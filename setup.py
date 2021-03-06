from setuptools import setup

setup(name='beamconv',
      description='Code for beam convolution.',
      url='https://github.com/oskarkleincentre/cmb_beams',
      author='Adri J. Duivenvoorden',
      author_email='adri.j.duivenvoorden@gmail.com',
      license='MIT',
      packages=['beamconv'],
      extras_require={'test' : ['pytest']},
      zip_safe=False)
