#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# built-in modules

# other modules
import numpy as np
from scipy.integrate import odeint

__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'


def simulate_classical(state0, times, DV):
    # Perform classical simulation
    states = odeint(lambda y, t: DV @ y, state0, times)
    return states
