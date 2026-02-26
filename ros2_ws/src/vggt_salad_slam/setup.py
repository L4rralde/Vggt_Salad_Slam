from setuptools import find_packages, setup

package_name = 'vggt_salad_slam'

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
    maintainer='emmanuel',
    maintainer_email='ealarralde@gmail.com',
    description='VGGT-SALAD Like models interface',
    license='Apache 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'main = vggt_salad_slam.vggt_salad_slam:main'
        ],
    },
)
