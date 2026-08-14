import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'item_finder'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name,
            ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'models'),
            glob(os.path.join('models', '*'))),
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='Arm Physical AI hackathon submission: on-device object search robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = item_finder.perception_node:main',
            'query_matcher_node = item_finder.query_matcher_node:main',
            'search_state_node = item_finder.search_state_node:main',
        ],
    },
)