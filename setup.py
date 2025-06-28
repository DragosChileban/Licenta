from setuptools import setup

setup(
   name='Licenta',
   version='1.0',
   description='CrashSplat: 2D to 3D Vehicle Damage Detection in Gaussian Splatting',
   author='Dragos-Andrei Chileban',
   author_email='dragos-andrei.chileban@student.upt.ro',
   packages=['Licenta'],  
#    install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
   scripts=[
            'segmentation/run_script.py',
           ]
)