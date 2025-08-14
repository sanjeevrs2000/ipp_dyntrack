from setuptools import find_packages, setup

package_name = 'dyntrack_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/dyn_map.launch.py']),
        ('share/' + package_name + '/launch', ['launch/informative_planner_log.launch.py']),
        ('share/' + package_name + '/launch', ['launch/baseline_log.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sanjeev',
    maintainer_email='sanjeevrs2000@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'occupancy_grid = dyntrack_planner.grid_map:main',
            'informative_planner = dyntrack_planner.informative_planner:main',
            'data_logger = dyntrack_planner.logger:main',
            'baseline_planner = dyntrack_planner.baseline_planner:main',
            'perception = dyntrack_planner.perception:main',
        ],
    },
)
