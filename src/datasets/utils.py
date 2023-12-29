import hashlib
import urllib
import urllib.parse
import urllib.request
from pathlib import Path

import requests
from tqdm import tqdm


def download_url(url, path, chunk_size=65536, check_certificate=True, overwrite=False):
    path = Path(path)

    if path.is_dir():
        path = path / urllib.parse.unquote(url.split("/")[-1])

    print(f"Downloading {url} to {path}...")
    response = requests.get(url, stream=True, verify=check_certificate)
    total_size = int(response.headers.get("content-length", 0))
    file_size = path.stat().st_size if path.is_file() else None

    if not overwrite and file_size == total_size:
        return path

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(path, "wb") as f:
            for data in response.iter_content(chunk_size):
                progress_bar.update(len(data))
                f.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")

    return path


def hash_file(path, method="sha256", bufsize=131072):
    hash = hashlib.sha256() if method == "sha256" else None
    mv = memoryview(bytearray(bufsize))
    with open(path, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            hash.update(mv[:n])
    return hash.hexdigest()
