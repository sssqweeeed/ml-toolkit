import json
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import urlopen


def main() -> None:
    url, directory = sys.argv[1:]
    expected = json.loads((Path(directory) / "versions.json").read_text())
    if not url.startswith("https://"):
        raise ValueError("Expected the public HTTPS documentation URL")
    for attempt in range(30):
        try:
            suffix = f"?verify={time.time_ns()}"
            with urlopen(
                urljoin(url, "versions.json") + suffix, timeout=20
            ) as response:
                actual = json.load(response)
            if actual != expected:
                raise ValueError("Public version list does not match the uploaded site")
            for version in expected:
                for name in [version["version"], *version["aliases"]]:
                    with urlopen(
                        urljoin(url, f"{name}/") + suffix, timeout=20
                    ) as response:
                        if response.status != 200:
                            raise ValueError(f"Documentation unavailable: {name}")
            print("Public version list and documentation channels verified")
            return
        except (URLError, ValueError, TimeoutError) as error:
            if attempt == 29:
                raise
            print(f"Waiting for public documentation: {error}", flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
