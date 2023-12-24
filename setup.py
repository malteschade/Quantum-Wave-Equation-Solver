#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Setup script for the quantum 1D elastic wave equation solver module.}

{
    Copyright (C) [2023]  [Malte Schade]

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
}
"""

# -------- IMPORTS --------
# Built-in modules
from setuptools import setup, find_packages

# -------- SETUP --------
with open('requirements.txt', encoding='utf8') as f:
    required = f.read().splitlines()

setup(
    name='qcws',
    version='1.0.0',
    packages=find_packages(),
    install_requires=required,
    author='Malte Leander Schade',
    author_email='mail@malteschade.com',
    description='',
    url='https://github.com/malteschade/Quantum-Wave-Equation-Solver',
    license='GPLv3',
    status='IN PROGRESS'
)
