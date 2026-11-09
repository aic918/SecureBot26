"""Start / Stop window for room laps.

Publishes "start" / "stop" on /laps/command and shows /laps/status (which room, which round
and loop, how much of the floor one round sweeps). Closing the window also sends "stop".
"""
import json
import signal
import threading
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

STATE_TEXT = {'idle': 'Stopped', 'running': 'Running laps', 'done': 'Done',
              'error': 'Cannot start', 'home': 'Returning home'}
STATE_COLOR = {'done': '#1a7f37', 'error': '#cf222e', 'idle': '#57606a'}


class PanelNode(Node):
    def __init__(self):
        super().__init__('laps_panel')
        self.pub = self.create_publisher(String, '/laps/command', 10)
        self.status = None
        self.create_subscription(String, '/laps/status', self.status_cb, 10)

    def status_cb(self, msg):
        try:
            self.status = json.loads(msg.data)
        except ValueError:
            pass

    def send(self, command):
        self.pub.publish(String(data=command))


def main(args=None):
    rclpy.init(args=args)
    node = PanelNode()
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    root = tk.Tk()
    root.title('SecureBot Laps')
    root.geometry('+60+60')
    font = ('Helvetica', 20, 'bold')
    room_label = tk.Label(root, text='Waiting for the laps node...',
                          font=('Helvetica', 16, 'bold'), anchor='w', padx=16, pady=8)
    room_label.pack(fill='x')
    buttons = tk.Frame(root, padx=16)
    buttons.pack(fill='x')
    for text, cmd, bg, active in (('START', 'start', '#2da44e', '#2c974b'),
                                  ('STOP', 'stop', '#cf222e', '#a40e26'),
                                  ('RETURN\nHOME', 'home', '#0969da', '#0550ae')):
        tk.Button(buttons, text=text, font=font, bg=bg, fg='white', activebackground=active,
                  width=8, height=2, command=lambda c=cmd: node.send(c)).pack(side='left', padx=8)

    # no-pass lines: drawn with RViz's "Publish Point" tool while drawing is on
    lines = tk.Frame(root, padx=16, pady=8)
    lines.pack(fill='x')
    small = ('Helvetica', 12, 'bold')
    draw_button = tk.Button(lines, text='Draw no-pass line', font=small, width=16,
                            command=lambda: node.send('nopass draw'))
    draw_button.pack(side='left', padx=(8, 4))
    tk.Button(lines, text='Undo line', font=small,
              command=lambda: node.send('nopass undo')).pack(side='left', padx=4)
    tk.Button(lines, text='Clear lines', font=small,
              command=lambda: node.send('nopass clear')).pack(side='left', padx=4)
    lines_label = tk.Label(root, text='', font=('Helvetica', 11), anchor='w', padx=24,
                           justify='left', wraplength=440)
    lines_label.pack(fill='x')

    info = tk.Frame(root, padx=16, pady=8)
    info.pack(fill='x')
    state_label = tk.Label(info, text='', font=('Helvetica', 16, 'bold'), anchor='w')
    state_label.pack(fill='x')
    message_label = tk.Label(info, text='', font=('Helvetica', 12), anchor='w', wraplength=420,
                             justify='left')
    message_label.pack(fill='x', pady=(2, 8))
    plan_label = tk.Label(info, text='', font=('Helvetica', 12), anchor='w', justify='left')
    plan_label.pack(fill='x')
    bar = ttk.Progressbar(info, length=420, maximum=100)
    bar.pack(fill='x', pady=(2, 16))

    def refresh():
        s = node.status
        if s:
            state = s.get('state', '?')
            room_label.config(text=s.get('room', ''))
            state_label.config(text=STATE_TEXT.get(state, state),
                               fg=STATE_COLOR.get(state, '#0969da'))
            message_label.config(text=s.get('message', ''))
            loops, loop = s.get('loops', 0), s.get('loop', 0)
            text = (f'{loops} loops, {s.get("length", 0):.0f} m a round, sweeps '
                    f'{s.get("coverage", 0):.0%} of the floor')
            if s.get('remaining') is not None:
                text += f'\nThis loop: {s["remaining"]:.1f} m to go'
            plan_label.config(text=text)
            bar['value'] = 100 * (loop - 1) / loops if loops and loop else 0
            n = s.get('nopass', 0)
            if s.get('drawing'):
                draw_button.config(text='Finish line', bg='#1f2328', fg='white')
                lines_label.config(text='Drawing: in RViz pick "Publish Point" and click the '
                                        'corners of the line. Press Finish line when done.')
            else:
                draw_button.config(text='Draw no-pass line', bg='#f6f8fa', fg='black')
                lines_label.config(text=f'{n} no-pass line(s). Click a line or an orange box '
                                        'with "Publish Point" to remove it.')
        root.after(250, refresh)

    def on_close():
        node.send('stop')
        root.after(200, root.destroy)

    root.protocol('WM_DELETE_WINDOW', on_close)
    # Same as explore_panel: Ctrl+C / kill from ros2 launch must close the window too.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: root.after(0, on_close))
    refresh()
    try:
        root.mainloop()
    finally:
        node.send('stop')
        time.sleep(0.2)                   # let the stop go out
        rclpy.try_shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()


if __name__ == '__main__':
    main()
