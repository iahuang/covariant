"""
County Classes

Module Description
==================
This module contains the CountyData and CountyDataset classes.

Copyright Information
===============================
This file is Copyright (c) 2021 Deon Chan, Ian Huang, Emily Wan, Angela Zhuo.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Union
import json

from core.fs import read_file, write_file

ALPHANUM = "abcdefghijklmnopqrstuvwxyz0123456789"


def _condense_string(s: str) -> str:
    """
    "Condense" a string by making it lower case and removing all non-alphanumeric characters.
    """

    return "".join(c for c in s.lower() if c in ALPHANUM)


class CountyData:
    name: str
    state_abbv: str
    population: int
    land_area: float  # in square miles
    gdp: int  # in chained 2012 USD
    vacc_rate: float  # in a value from 0 to 1

    # Lat/Long coodinates
    geo_lat: float
    geo_long: float

    # COVID data
    # a long list of numbers where the index of each element
    # corresponds to a date index.
    cases: list[int]
    deaths: list[int]

    # Measured in 7 day rolling averages
    _cases_per_day: list[float]
    _deaths_per_day: list[float] 
    _avgs_computed_yet: bool

    def __init__(
        self,
        name: str,
        state_abbv: str,
        population: int,
        land_area: float,
        gdp: int,
        vacc_rate: float,
        geo_lat: float,
        geo_long: float,
        cases: list[int],
        deaths: list[int]
    ):
        self.name = name
        self.state_abbv = state_abbv
        self.population = population
        self.land_area = land_area
        self.gdp = gdp
        self.vacc_rate = vacc_rate
        self.geo_lat = geo_lat
        self.geo_long = geo_long
        self.cases = cases
        self.deaths = deaths
        self._cases_per_day = []
        self._deaths_per_day = []
        self._avgs_computed_yet = False

    def to_json(self) -> dict[str, Any]:
        """Return a JSON serializable copy of this object"""

        # Very quick and dirty method of serializing all members of this object
        # whose names are not prefixed by an underscore.
        return {k: v for (k, v) in self.__dict__.items() if not k.startswith("_")}

    def get_avg_cases_per_day(self) -> list[float]:
        if not self._avgs_computed_yet:
            self._compute_case_death_rates()
        return self._cases_per_day

    def get_avg_deaths_per_day(self) -> list[float]:
        if not self._avgs_computed_yet:
            self._compute_case_death_rates()
        return self._deaths_per_day

    def _compute_rates(self, metric_list: list[int], rate_list: list[float]) -> None:
        """
        NOTE: THIS DATA IS NO LONGER USED IN THE FINAL PROJECT. SOME CODE IS COMMENTED OUT
        FOR PERFORMANCE REASONS
        """
        # each item in this list is the number of new cases/deaths per day
        # i.e. the difference between the cases/deaths each day and the previous one.
        # For some reason, sometimes the number of total cases/deaths goes *down* from
        # the previous day in the dataset. For this, we'll clip the value
        # to a minimum of zero.
        _deltas = []

        # for i, value in enumerate(metric_list):
        #     if i == 0:
        #         _deltas.append(value)
        #     else:
        #         _deltas.append(max(value - metric_list[i-1], 0))

        # # Compute rolling averages

        # for i in range(len(_deltas)):
        #     previous_deltas = _deltas[max(i-8, 0): i+1]
        #     avg = sum(previous_deltas) / len(previous_deltas)

        #     rate_list.append(avg)

    def _compute_case_death_rates(self) -> None:
        """
        Compute average case/death rates per day based on a 7-day rolling average.
        """
        self._compute_rates(self.cases, self._cases_per_day)
        self._compute_rates(self.deaths, self._deaths_per_day)

        self._avgs_computed_yet = True


class CountyDataset:
    """A class used for quickly storing and indexing US counties"""

    _id_table: dict[str, CountyData]
    _state_table: dict[str, list[str]]

    # Date data
    dates: list[str]

    def __init__(self):
        # Stores a county "ID" to its instance
        self._id_table = {}
        # Stores states to a list of ids
        self._state_table = {}

        self.dates = []

    def iter(self) -> Iterable[CountyData]:
        """Return an iterable of the counties in this dataset"""

        return self._id_table.values()

    def has_county(self, county: CountyData) -> bool:
        return self._make_id(county) in self._id_table

    def add_county(self, county: CountyData, ignore_duplicates: bool = False) -> None:
        """Add a county to the dataset. Throw an error if it exists already"""

        if self.has_county(county) and not ignore_duplicates:
            raise ValueError("County "+repr(county) +
                             " already exists in the database!")

        cid = self._make_id(county)
        self._id_table[cid] = county

        if not county.state_abbv in self._state_table:
            self._state_table[county.state_abbv] = []

        self._state_table[county.state_abbv].append(cid)

    def remove_county(self, county: CountyData) -> None:
        """Remove a county from this dataset"""

        cid = self._make_id(county)
        self._id_table.pop(cid)
        self._state_table[county.state_abbv].remove(cid)

    def get_county(self, county_name: str, state_abbv: str) -> Union[CountyData, None]:
        """
        Get a county instance in the database given
        using its county name and its state's abbrevation. Return None if it doesn't exist
        """

        cid = _condense_string(county_name + state_abbv)

        return self._id_table.get(cid)

    def search_for_county(self, county_query: str, state_abbv: str) -> Union[CountyData, None]:
        """
        Search for a county instance in the database given
        a county name and its state's abbrevation. Return a best match or None if no match was found
        """

        cid = _condense_string(county_query)

        for name, county in self._id_table.items():
            if name.startswith(cid) and state_abbv == county.state_abbv:
                return self._id_table[name]

        return None

    def _make_id(self, county: CountyData) -> str:
        """Return a database ID from a county instance"""

        return self._id_from_county_info(county.name, county.state_abbv)

    def _id_from_county_info(self, name: str, state_abbv: str) -> str:
        return _condense_string(name + state_abbv)

    def save_dataset(self, path: str, dates: list[str]) -> None:
        """Save this dataset to a file"""

        write_file(path, json.dumps(
            {
                "counties": [county.to_json() for county in self._id_table.values()],
                "dates": dates
            }
        ))

    def load_dataset(self, path: str) -> None:
        """
        Load this dataset object with data from the specified file.
        """

        json_data = json.loads(read_file(path))
        county_data: list[dict[str, Any]] = json_data["counties"]

        for county in county_data:
            name = county["name"]
            abbv = county["state_abbv"]
            pop = county["population"]
            land_area = county["land_area"]
            gdp = county["gdp"]
            vacc_rate = county["vacc_rate"]
            geo_lat = county["geo_lat"]
            geo_long = county["geo_long"]
            cases = county["cases"]
            deaths = county["deaths"]

            county_instance = CountyData(
                name,
                abbv,
                pop,
                land_area,
                gdp,
                vacc_rate,
                geo_lat,
                geo_long,
                cases,
                deaths
            )

            self.add_county(county_instance)

        self.dates = json_data["dates"]

    def get_length(self) -> int:
        """Return the number of counties in this dataset"""

        return len(self._id_table)

    def get_counties(self) -> Iterable[CountyData]:
        return (c for c in self._id_table.values())


def state_name_to_abbv(name: str) -> str:
    """Return a state's abbreviation based on its name"""

    return _NAME_TO_ABBV[_condense_string(name)]


def abbv_to_state_name(abbv: str) -> str:
    """REturn a state's name based on its abbrevation"""

    return _ABBV_TO_NAME[abbv]


_NAME_TO_ABBV = {_condense_string(k): v for (k, v) in {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District Of Columbia': 'DC',
    'Federated States Of Micronesia': 'FM',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Marshall Islands': 'MH',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands': 'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Palau': 'PW',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}.items()}

_ABBV_TO_NAME = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AS': 'American Samoa',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'DC': 'District Of Columbia',
    'FM': 'Federated States Of Micronesia',
    'FL': 'Florida',
    'GA': 'Georgia',
    'GU': 'Guam',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MH': 'Marshall Islands',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'MP': 'Northern Mariana Islands',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PW': 'Palau',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VI': 'Virgin Islands',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming'
}
