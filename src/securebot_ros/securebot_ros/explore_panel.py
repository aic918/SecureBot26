"""Start / Stop window for room exploration.

Publishes "start" / "stop" on /explore/command and shows /explore/status (state, what the
robot is doing, how much of the room is scanned). Closing the window also sends "stop".
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

STATE_TEXT = {
    'idle': 'Stopped', 'survey': 'Looking around', 'plan': 'Choosing a direction',
    'turn': 'Turning', 'drive': 'Driving', 'pause': 'Pausing', 'blocked_wait': 'Blocked',
    'backup': 'Backing up', 'done': 'Done', 'error': 'Stopped (problem)',
}
STATE_COLOR = {'done': '#1a7f37', 'error': '#cf222e', 'blocked_wait': '#bc4c00',
               'backup': '#bc4c00', 'idle': '#57606a'}


class PanelNode(Node):
    def __init__(self):
        super().__init__('explore_panel')
        self.pub = self.create_publisher(String, '/explore/command', 10)
        self.status = None
        self.create_subscription(String, '/explore/status', self.status_cb, 10)

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
    root.title('SecureBot Explore')
    root.geometry('+60+60')
    font = ('Helvetica', 20, 'bold')
    buttons = tk.Frame(root, padx=16, pady=16)
    buttons.pack(fill='x')
    tk.Button(buttons, text='START', font=font, bg='#2da44e', fg='white', activebackground='#2c974b',
              width=8, height=2, command=lambda: node.send('start')).pack(side='left', padx=8)
    tk.Button(buttons, text='STOP', font=font, bg='#cf222e', fg='white', activebackground='#a40e26',
              width=8, height=2, command=lambda: node.send('stop')).pack(side='left', padx=8)

    info = tk.Frame(root, padx=16, pady=4)
    info.pack(fill='x')
    state_label = tk.Label(info, text='Waiting for the explore node...', font=('Helvetica', 16, 'bold'),
                           anchor='w')
    state_label.pack(fill='x')
    message_label = tk.Label(info, text='', font=('Helvetica', 12), anchor='w', wraplength=420,
                             justify='left')
    message_label.pack(fill='x', pady=(2, 8))
    coverage_label = tk.Label(info, text='Room scanned: -', font=('Helvetica', 12), anchor='w')
    coverage_label.pack(fill='x')
    bar = ttk.Progressbar(info, length=420, maximum=100)
    bar.pack(fill='x', pady=(2, 16))

    def refresh():
        s = node.status
        if s:
            state = s.get('state', '?')
            state_label.config(text=STATE_TEXT.get(state, state),
                               fg=STATE_COLOR.get(state, '#0969da'))
            message_label.config(text=s.get('message', ''))
            cov, goal = s.get('coverage'), s.get('goal', 0.9)
            if cov is None:
                coverage_label.config(text=f'Room scanned: - (goal {goal:.0%})')
                bar['value'] = 0
            else:
                coverage_label.config(text=f'Room scanned: {cov:.0%} (goal {goal:.0%}), '
                                           f'legs driven: {s.get("legs", 0)}')
                bar['value'] = cov * 100
        root.after(250, refresh)

    def on_close():
        node.send('stop')
        root.after(200, root.destroy)

    root.protocol('WM_DELETE_WINDOW', on_close)
    # Ctrl+C / kill (e.g. from ros2 launch) must close the window too; rclpy's own handlers only
    # shut ROS down and would leave the window open. refresh() runs Python every 250 ms, so the
    # handler gets a chance to run inside Tk's main loop.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: root.after(0, on_close))
    refresh()
    try:
        root.mainloop()
    finally:
        node.send('stop')
        time.sleep(0.2)                   # let the stop go out
        rclpy.try_shutdown()              # ends rclpy.spin in the background thread first
        spinner.join(timeout=2.0)
        node.destroy_node()


if __name__ == '__main__':
    main()
