from setuptools import find_packages, setup


def get_requirements(file_path):
    '''
    This function will return a list of requrirements/libraires
    '''

    with open(file_path) as file:
        req = file.readlines()
        req = [r.replace('\n', '')  for r in req]

    if '-e .' in req:
        req.remove('-e .')

    return req




setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Manan',
    author_email= 'mananvijay72@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')


)