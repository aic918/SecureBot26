from setuptools import find_packages, setup

package_name = 'securebot_base'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='securebot',
    maintainer_email='securebot@example.com',
    description='SecureBot base package for ESP32 serial bridge',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'esp32_serial_bridge = securebot_base.esp32_serial_bridge:main',
        ],
    },
)
