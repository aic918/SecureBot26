import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'securebot_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='securebot',
    maintainer_email='aicano918@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },

entry_points={
    'console_scripts': [
         'securebot_live_node = securebot_ros.securebot_live_node:main',
        'enroll_faces = securebot_ros.enroll_faces:main',
        'yolo_person_node = securebot_ros.yolo_person_node:main',
        'motion_manager_node = securebot_ros.motion_manager_node:main',
        'simple_patrol_node = securebot_ros.simple_patrol_node:main',
    ],
},

)
