#!/usr/bin/env bash
sudo rm -r dist/ build/ arbok.egg-info/
python3 setup.py sdist bdist_wheel
twine upload dist/*