"""SecureBot Rooms window: name the rooms the robot has mapped, and look at the saved maps.

  ros2 run securebot_ros room_manager                    # maps in ~/securebot_ws/maps
  ros2 run securebot_ros room_manager --maps-dir <dir>

Every saved LiDAR map (the .yaml + .pgm pairs map_saver_cli writes) is listed; give one a room
name, rename a room, point it at a newer map, or remove the name (the files stay). Selecting a
row shows that map. "Save current map..." saves the map SLAM is building right now (the explore
or SLAM launch must be running) and names it in one go. Names are kept in <maps-dir>/rooms.json
(see room_store). Needs no ROS itself, except for saving a new map.
"""
import argparse
import os
import shutil
import signal
import subprocess
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, simpledialog, ttk

from securebot_ros import room_store as rs

SLAM_PROCESS = 'async_slam_tool'    # async_slam_toolbox_node, as the kernel names it (15 chars)


def slam_running():
    """slam_toolbox (explore / SLAM launch) is running on this machine."""
    return subprocess.run(['pgrep', '-x', SLAM_PROCESS],
                          capture_output=True).returncode == 0


ZOOM_LEVELS = [(1, 8), (1, 6), (1, 4), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (6, 1),
               (8, 1)]                   # (zoom, subsample): Tk scales images by whole numbers
FIT = 5                                  # index of 1:1
UNNAMED = '(no name)'
MISSING = '(map file missing)'


class RoomManager:
    def __init__(self, root, maps_dir):
        self.root = root
        self.store = rs.RoomStore(maps_dir)
        self.rows = {}                   # tree item id -> (room name or None, MapInfo or None)
        self.image = None                # the selected map at 1:1
        self.shown = None                # the scaled copy on the canvas (Tk needs a reference)
        self.zoom = FIT
        self.saving = None               # (room name, stem, result dict) while map_saver runs

        root.title('SecureBot Rooms')
        # Sizes follow the font, so the window also fits on a high-DPI screen with big fonts.
        line = tkfont.nametofont('TkDefaultFont').metrics('linespace')
        ttk.Style(root).configure('Treeview', rowheight=line + 6)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f'{min(line * 60, int(sw * 0.9))}x{min(line * 30, int(sh * 0.85))}+60+60')
        root.minsize(min(line * 40, sw), min(line * 20, sh))
        top = tk.Frame(root, padx=10, pady=8)
        top.pack(fill='x')
        tk.Label(top, text='Rooms and saved maps',
                 font=('Helvetica', 16, 'bold')).pack(anchor='w')
        tk.Label(top, text=f'Folder: {maps_dir}', fg='#57606a', anchor='w').pack(fill='x')

        body = ttk.PanedWindow(root, orient='horizontal')
        body.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        left = tk.Frame(body)
        right = tk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        # --- list of rooms / maps ---
        cols = ('room', 'map', 'saved', 'size')
        self.tree = ttk.Treeview(left, columns=cols, show='headings', selectmode='browse')
        for c, text, chars in (('room', 'Room', 12), ('map', 'Map file', 16),
                               ('saved', 'Saved', 11), ('size', 'Size (m)', 9)):
            self.tree.heading(c, text=text)
            self.tree.column(c, width=line * chars // 2, anchor='w')
        self.tree.tag_configure('unnamed', foreground='#57606a')
        self.tree.tag_configure('missing', foreground='#cf222e')
        self.tree.pack(fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', lambda _: self.show_selected())
        self.tree.bind('<Double-1>', lambda _: self.name_or_rename())

        bar = tk.Frame(left, pady=6)
        bar.pack(fill='x')
        for c in range(3):
            bar.columnconfigure(c, weight=1, uniform='buttons')
        for i, (text, cmd, color) in enumerate((
                ('Name this map...', self.add_room, None),
                ('Rename room...', self.rename_room, None),
                ('Change map...', self.change_map, None),
                ('Remove room name', self.remove_room, None),
                ('Delete map files...', self.delete_map, '#cf222e'),
                ('Refresh', self.refresh, None))):
            tk.Button(bar, text=text, command=cmd, pady=2, **({'fg': color} if color else {})) \
                .grid(row=i // 3, column=i % 3, sticky='we')
        self.save_button = tk.Button(bar, text='Save current map as a new room...',
                                     bg='#2da44e', fg='white', activebackground='#2c974b',
                                     command=self.save_current, pady=4)
        self.save_button.grid(row=2, column=0, columnspan=3, sticky='we', pady=(6, 0))
        tk.Button(bar, text='Clear live map (start mapping afresh)', command=self.clear_live,
                  pady=2).grid(row=3, column=0, columnspan=3, sticky='we', pady=(4, 0))
        self.status = tk.Label(left, text='', anchor='w', justify='left')
        self.status.pack(fill='x')
        self.status.bind('<Configure>', lambda e: self.status.config(wraplength=e.width))

        # --- map view ---
        tools = tk.Frame(right)
        tools.pack(fill='x')
        self.title = tk.Label(tools, text='Select a map', font=('Helvetica', 14, 'bold'),
                              anchor='w')
        self.title.pack(side='left', fill='x', expand=True)
        for text, cmd in (('Fit', self.fit), ('-', lambda: self.step_zoom(-1)),
                          ('+', lambda: self.step_zoom(1))):
            tk.Button(tools, text=text, width=3, command=cmd).pack(side='left', padx=1)
        tk.Button(tools, text='Copy nav command', command=self.copy_nav).pack(side='left', padx=4)
        view = tk.Frame(right)
        view.pack(fill='both', expand=True, pady=4)
        self.canvas = tk.Canvas(view, bg='#808080', highlightthickness=0)
        ys = ttk.Scrollbar(view, orient='vertical', command=self.canvas.yview)
        xs = ttk.Scrollbar(view, orient='horizontal', command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        ys.grid(row=0, column=1, sticky='ns')
        xs.grid(row=1, column=0, sticky='ew')
        view.rowconfigure(0, weight=1)
        view.columnconfigure(0, weight=1)
        self.canvas.bind('<ButtonPress-1>', lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind('<B1-Motion>', lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.details = tk.Label(right, text='', anchor='w', justify='left', fg='#57606a',
                                font=('Helvetica', 10))
        self.details.pack(fill='x')
        self.details.bind('<Configure>', lambda e: self.details.config(wraplength=e.width))
        tk.Label(right, text='White = free, black = wall, gray = not seen. Drag to pan.',
                 anchor='w', fg='#57606a', font=('Helvetica', 9)).pack(fill='x')

        self.refresh()

    # --- helpers ------------------------------------------------------------------------
    def say(self, text, error=False):
        self.status.config(text=text, fg='#cf222e' if error else '#1a7f37')

    def selected(self):
        sel = self.tree.selection()
        return self.rows.get(sel[0]) if sel else None

    def run(self, action, done):
        """Run a store change; show its RoomError instead of crashing the window."""
        try:
            action()
        except rs.RoomError as e:
            messagebox.showerror('SecureBot Rooms', str(e), parent=self.root)
            return
        except OSError as e:
            messagebox.showerror('SecureBot Rooms', f'Could not write the files: {e}',
                                 parent=self.root)
            return
        self.refresh()
        self.say(done)

    def ask_name(self, title, prompt, initial=''):
        return simpledialog.askstring(title, prompt, initialvalue=initial, parent=self.root)

    def refresh(self, select_stem=None, select_name=None):
        keep = self.selected()
        if keep and not (select_stem or select_name):
            select_name, select_stem = keep[0], keep[1].stem if keep[1] else None
        try:
            self.store.load()
        except rs.RoomError as e:
            self.say(str(e), error=True)
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        target = by_stem = None          # keep the selection by name, else by map (after a rename)
        for name, info in self.store.entries():
            if info:
                w, h = info.size_m
                values = (name or UNNAMED, info.stem,
                          time.strftime('%b %d %H:%M', time.localtime(info.modified)),
                          f'{w:.1f} x {h:.1f}')
            else:
                values = (name, MISSING, '', '')
            tag = 'missing' if not info else ('' if name else 'unnamed')
            iid = self.tree.insert('', 'end', values=values, tags=(tag,))
            self.rows[iid] = (name, info)
            if select_name and name == select_name:
                target = iid
            elif select_stem and info and info.stem == select_stem and not by_stem:
                by_stem = iid
        target = target or by_stem
        if not self.rows:
            self.say('No saved maps yet. Explore a room, then "Save current map as a new room".')
        if target:
            self.tree.selection_set(target)
            self.tree.see(target)
        else:
            self.show_selected()

    # --- room actions -------------------------------------------------------------------
    def name_or_rename(self):
        row = self.selected()
        if row and row[0]:
            self.rename_room()
        else:
            self.add_room()

    def add_room(self):
        row = self.selected()
        if not row or not row[1]:
            self.say('Select a map in the list first.', error=True)
            return
        if row[0]:
            self.say(f'That map is already "{row[0]}". Use Rename to change the name.', error=True)
            return
        name = self.ask_name('Name this map', f'Room name for {row[1].stem}:')
        if name is not None:
            self.run(lambda: self.store.add(name, row[1].stem), f'Added room "{name.strip()}".')

    def rename_room(self):
        row = self.selected()
        if not row or not row[0]:
            self.say('Select a named room first (unnamed maps: "Name this map").', error=True)
            return
        new = self.ask_name('Rename room', f'New name for "{row[0]}":', row[0])
        if new is not None and new.strip() != row[0]:
            self.run(lambda: self.store.rename(row[0], new),
                     f'Renamed "{row[0]}" to "{" ".join(new.split())}".')

    def change_map(self):
        row = self.selected()
        if not row or not row[0]:
            self.say('Select a named room first.', error=True)
            return
        choices = [m for n, m in self.store.entries() if m and (n is None or n == row[0])]
        if not choices:
            self.say('There are no other unnamed maps to choose from.', error=True)
            return
        win = tk.Toplevel(self.root)
        win.title('Change map')
        win.transient(self.root)
        tk.Label(win, text=f'Map for "{row[0]}" (unnamed maps):', padx=10, pady=6).pack(anchor='w')
        box = tk.Listbox(win, width=50, height=min(12, len(choices)))
        for m in choices:
            saved = time.strftime('%b %d %H:%M', time.localtime(m.modified))
            box.insert('end', f'{m.stem}   ({saved})')
        box.pack(padx=10, fill='both', expand=True)

        def pick():
            sel = box.curselection()
            if sel:
                stem = choices[sel[0]].stem
                win.destroy()
                self.run(lambda: self.store.set_map(row[0], stem),
                         f'"{row[0]}" now uses {stem}.')
        box.bind('<Double-1>', lambda _: pick())
        tk.Button(win, text='Use this map', command=pick).pack(pady=8)
        win.grab_set()

    def remove_room(self):
        row = self.selected()
        if not row or not row[0]:
            self.say('Select a named room first.', error=True)
            return
        if messagebox.askyesno('Remove room', f'Remove the room name "{row[0]}"?\n\n'
                               'The map files are kept and show up as unnamed.', parent=self.root):
            self.run(lambda: self.store.remove(row[0]), f'Removed room "{row[0]}".')

    def delete_map(self):
        row = self.selected()
        if not row or not row[1]:
            self.say('Select a saved map first.', error=True)
            return
        info = row[1]
        what = f' (room "{row[0]}")' if row[0] else ''
        if messagebox.askyesno('Delete map files',
                               f'Permanently delete {info.stem}{what}?\n\n{info.yaml_path}\n'
                               f'{info.image_path}\n\nThis cannot be undone.',
                               icon='warning', parent=self.root):
            self.run(lambda: self.store.delete_map(info.stem), f'Deleted {info.stem}.')
            # SLAM still holds everything scanned since the launch: saving again would bring
            # the deleted map back (2026-11-07). Offer to clear that too.
            if slam_running() and messagebox.askyesno(
                    'Clear the live map too?',
                    'SLAM is still building a map that holds everything scanned since the '
                    'launch, including what you just deleted. Saving it later would bring that '
                    'back.\n\nClear the live map as well and start mapping afresh?',
                    parent=self.root):
                self.clear_live(ask=False)

    def clear_live(self, ask=True):
        """Restart slam_toolbox (the launch respawns it): the live map starts empty."""
        if not slam_running():
            self.say('SLAM is not running (no explore / SLAM launch): nothing to clear.',
                     error=True)
            return
        if ask and not messagebox.askyesno(
                'Clear live map', 'Throw away the map SLAM is building now and start mapping '
                'afresh from where the robot is?\n\nSaved maps are not touched. Keep the robot '
                'still for a few seconds.', parent=self.root):
            return
        subprocess.run(['pkill', '-INT', '-x', SLAM_PROCESS], check=False)
        self.say('Live map cleared: SLAM restarts in a few seconds with an empty map.')

    # --- saving the live map -------------------------------------------------------------
    def save_current(self):
        if self.saving:
            return
        if not shutil.which('ros2'):
            messagebox.showerror('SecureBot Rooms', 'ros2 is not on the PATH: source ROS 2 and '
                                 'the workspace first.', parent=self.root)
            return
        name = self.ask_name('Save current map', 'Room name for the map being built now:')
        if name is None:
            return
        try:
            name = rs.clean_name(name)
            if self.store.find(name):
                raise rs.RoomError(f'A room called "{name}" already exists. Remove or rename it '
                                   'first, or save under a new name.')
        except rs.RoomError as e:
            messagebox.showerror('SecureBot Rooms', str(e), parent=self.root)
            return
        stem = rs.stem_for(name)
        os.makedirs(self.store.maps_dir, exist_ok=True)
        cmd = ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f',
               os.path.join(self.store.maps_dir, stem), '--ros-args', '-p',
               'save_map_timeout:=10.0']
        result = {}

        def work():
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
                result['ok'] = p.returncode == 0
                result['log'] = (p.stdout + p.stderr).strip()
            except (OSError, subprocess.TimeoutExpired) as e:
                result['ok'], result['log'] = False, str(e)
            result['done'] = True
        self.saving = (name, stem, result)
        self.save_button.config(state='disabled', text='Saving the map...')
        self.say('Saving the map from SLAM (up to 10 s)...')
        threading.Thread(target=work, daemon=True).start()
        self.root.after(200, self.poll_save)

    def poll_save(self):
        name, stem, result = self.saving
        if not result.get('done'):
            self.root.after(200, self.poll_save)
            return
        self.saving = None
        self.save_button.config(state='normal', text='Save current map as a new room...')
        if not (result['ok'] and self.store.map_exists(stem)):
            last = '\n'.join(result.get('log', '').splitlines()[-6:])
            messagebox.showerror('SecureBot Rooms', 'The map was not saved. Is the explore or '
                                 f'SLAM launch running?\n\n{last}', parent=self.root)
            self.say('The map was not saved.', error=True)
            return
        try:
            self.store.add(name, stem)
        except rs.RoomError as e:
            self.say(f'Saved {stem}, but could not name it: {e}', error=True)
        else:
            self.say(f'Saved the map as room "{name}" ({stem}).')
        self.refresh(select_stem=stem)

    # --- map view -----------------------------------------------------------------------
    def show_selected(self):
        row = self.selected()
        self.canvas.delete('all')
        self.image = self.shown = None
        if not row:
            self.title.config(text='Select a map')
            self.details.config(text='')
            return
        name, info = row
        if not info:
            self.title.config(text=name)
            self.details.config(text='The map file for this room is missing. Use "Change map..." '
                                'to pick another, or "Remove room name".')
            return
        self.title.config(text=name or info.stem)
        try:
            self.image = tk.PhotoImage(file=info.image_path)
        except tk.TclError as e:
            self.details.config(text=f'Could not open {info.image_path}: {e}')
            return
        w, h = info.size_m
        self.details.config(text=(
            f'{info.stem}   {info.width} x {info.height} px at {info.resolution * 100:.0f} cm/px '
            f'= {w:.1f} x {h:.1f} m   origin ({info.origin[0]:.2f}, {info.origin[1]:.2f})\n'
            f'{info.yaml_path}'))
        self.fit()

    def fit(self):
        if not self.image:
            return
        self.canvas.update_idletasks()
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        iw, ih = self.image.width(), self.image.height()
        best = 0
        for i, (z, s) in enumerate(ZOOM_LEVELS):
            if iw * z / s <= cw and ih * z / s <= ch:
                best = i
        self.zoom = best
        self.draw()

    def step_zoom(self, d):
        if self.image:
            self.zoom = min(max(self.zoom + d, 0), len(ZOOM_LEVELS) - 1)
            self.draw()

    def draw(self):
        z, s = ZOOM_LEVELS[self.zoom]
        img = self.image.zoom(z) if z > 1 else self.image
        self.shown = img.subsample(s) if s > 1 else img
        self.canvas.delete('all')
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = max((cw - self.shown.width()) // 2, 0), max((ch - self.shown.height()) // 2, 0)
        self.canvas.create_image(x, y, image=self.shown, anchor='nw')
        self.canvas.configure(scrollregion=(0, 0, max(cw, x + self.shown.width()),
                                            max(ch, y + self.shown.height())))

    def copy_nav(self):
        row = self.selected()
        if not row or not row[1]:
            self.say('Select a saved map first.', error=True)
            return
        cmd = f'ros2 launch securebot_ros securebot_nav.launch.py map:={row[1].yaml_path}'
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.say(f'Copied: {cmd}')


def main(args=None):
    parser = argparse.ArgumentParser(description='Name rooms and view saved LiDAR maps.')
    parser.add_argument('--maps-dir',
                        default=os.environ.get('SECUREBOT_MAPS', rs.DEFAULT_MAPS_DIR),
                        help='folder with the saved maps (default %(default)s)')
    # ros2 launch appends --ros-args ...; ignore anything we do not know.
    opts, _ = parser.parse_known_args(args)
    root = tk.Tk()
    RoomManager(root, os.path.abspath(os.path.expanduser(opts.maps_dir)))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: root.after(0, root.destroy))
    tick = lambda: root.after(250, tick)  # noqa: E731  lets Python see Ctrl+C inside Tk's loop
    tick()
    root.mainloop()


if __name__ == '__main__':
    main()
