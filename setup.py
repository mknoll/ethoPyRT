from distutils.core import setup
setup(
  name = 'ethoPy',   
  packages = ['ethoPy'],   
  version = '0.1', 
  license='MIT',       
  description = 'Helper function for analysis of Varian ETHOS data',   
  author = 'Maximilian Knoll',                   
  author_email = 'm.knoll@dkfz.de',      
  url = 'https://github.com/mknoll/ethoPy',   
  download_url = 'https://github.com/mknoll/ethoPy/archive/refs/tags/v_01.tar.gz',  
  keywords = ['Radiotherapy', 'ETHOS'],   
  install_requires=[            
          'nibabel',
          'SimpleITK',
          'dcmrtstruct2nii',
          'logging',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',   

    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',

    'Programming Language :: Python :: 3', 
    'Programming Language :: Python :: 3.11',
  ],
)
