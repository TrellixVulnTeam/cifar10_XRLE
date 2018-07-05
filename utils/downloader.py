import tarfile
import os

import requests
import click
from tqdm import tqdm


def get_data(url, destination='./datasets'):
    _get_data(
        extract_only=False,
        remove_tar=True,
        destination=destination,
        url=url
    )


@click.command()
@click.option('--extract-only', '-e', is_flag=True)
@click.option('--remove-tar', '-r', is_flag=True)
@click.option('--destination', '-d', default='./datasets')
@click.option('--url', '-u', default='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
def _main(extract_only, remove_tar, destination, url):
    _get_data(
        extract_only=extract_only,
        remove_tar=remove_tar,
        destination=destination,
        url=url
    )


def _get_data(extract_only, remove_tar, destination, url):
    filename = os.path.basename(url)
    tar_path = os.path.join(destination, filename)

    if not extract_only:
        _download_data(url, tar_path)

    _extract_data(tar_path)

    if remove_tar:
        os.remove(tar_path)


def _download_data(url, destination_path):
    r = requests.get(url, stream=True)
    file_size = int(r.headers.get('Content-Length'))

    with open(destination_path, 'wb') as handler:
        for data in tqdm(r.iter_content(), total=file_size):
            handler.write(data)


def _extract_data(source_file):
    with tarfile.open(source_file, 'r:gz') as archive:
        archive.extractall()


if __name__ == '__main__':
    _main()
