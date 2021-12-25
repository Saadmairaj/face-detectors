from setuptools import setup, find_packages


def get_long_description(path):
    """Opens and fetches text of long descrition file."""
    with open(path, 'r') as f:
        return f.read()


attrs = dict(
    name='face-detectors',
    version="0.1.0",
    packages=find_packages(exclude=('test',)),
    long_description=get_long_description('README.md'),
    description='Light weight face detector hight level client with multiple detection techniques.',
    long_description_content_type='text/markdown',
    author="Saad Mairaj",
    author_email='Saadmairaj@yahoo.in',
    url='https://github.com/Saadmairaj/face-detectors',
    license='Apache',
    python_requires='>=3.6',
    setup_requires=[
        "cmake==3.21.1.post1",
    ],
    install_requires=[
        "dlib==19.22.0",
        "onnxoptimizer==0.2.6",
        "onnxruntime==1.8.1",
        "opencv-python==4.5.3.56",
        "opencv-contrib-python==4.5.3.56",
        "tqdm==4.62.1",
        "requests==2.26.0"
    ],
    classifiers=[
        'Environment :: MacOS X',
        'Environment :: Win32 (MS Windows)',
        'Environment :: X11 Applications',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Widget Sets',
        'License :: OSI Approved :: Apache Software License',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/Saadmairaj/face-detectors/issues',
        'Source': 'https://github.com/Saadmairaj/face-detectors',
        'Documentation': 'https://github.com/Saadmairaj/face-detectors#documentation',
    },
    keywords=[
        'machine learning',
        'face',
        'detector',
        'face detection',
        'CNN',
        'dlib',
        'ultrafast',
        'HOG',
        'caffemodel',
    ],
    include_package_data=True,
)

setup(**attrs)
