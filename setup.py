from setuptools import setup

setup(
   name='CrashSplat',
   version='1.0',
   description='CrashSplat: 2D to 3D Vehicle Damage Detection in Gaussian Splatting',
   author='Dragos-Andrei Chileban',
   author_email='dragos-andrei.chileban@student.upt.ro',
   packages=['CrashSplat'],  
#    install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
   scripts=[
            'scripts/run_script.py',
           ]
)