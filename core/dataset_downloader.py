"""
Dataset Downloader

Module Description
==================
Module for downloading and building the project database.

Copyright Information
===============================
This file is Copyright (c) 2021 Deon Chan, Ian Huang, Emily Wan, Angela Zhuo.
"""

from typing import Union
from termcolor import colored
from core.util import truncate, user_warn, unwrap
import core.fs as fs
from core.logger import info, panic, warn, complete_done, complete_err
import os
import requests
import zipfile
from core.county import CountyData, CountyDataset, state_name_to_abbv
import csv
import sys
import json
import re
import time

"""
US Gazetter Database

Contains geogrpahic data per county on land/water surface area as well as geographic location
given in long/lat coordinates.
"""
URL_GAZETTER = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip"

"""
US Census Database

Contains population estimates for the US 2020 Census by county
"""
URL_CENSUS = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020.csv"

"""
US Vaccination Data

Contains data per county on COVID-19 vaccination
"""
URL_VACCINATION = "https://data.cdc.gov/api/views/8xkx-amqh/rows.csv?accessType=DOWNLOAD"

"""
US BEA Dataset

Contains data per county on GDP
"""
BEA_KEY = '437EB18F-A623-44AE-A145-7A40DB58DB8C'
BEA_DATASETLIST = f'https://apps.bea.gov/api/data/?&UserID={BEA_KEY}&method=GETDATASETLIST&ResultFormat=JSON'
GDP_PARAMS = f'https://apps.bea.gov/api/data/?&UserID={BEA_KEY}&method=GETPARAMETERLIST&datasetname=Regional&ResultFormat=JSON'
URL_GDP = f'https://apps.bea.gov/api/data/?&UserID={BEA_KEY}&method=GETDATA&datasetname=Regional&Year=2019&TableName=CAGDP1&LineCode=1&GeoFips=COUNTY&ResultFormat=JSON'

"""
NY Times COVID-19 Dataset
"""
URL_COVID = "https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv"


def _download_file(url: str, path: str, overwrite: bool = True) -> None:
    """
    Downloads a file at the given URL to be saved to the given path.
    If a file exists at the given path, and overwrite=False, then
    don't redownload the file.
    """

    # If the overwrite flag is set to false, and the file exists
    # then don't redownload it.
    if os.path.exists(path) and not overwrite:
        info("Skipped downloading file at url \"{}\"".format(
            truncate(url, 64),
        ))
        return

    # Log
    info("Downloading file at url \"{}\"".format(
        truncate(url, 64)
    ), incomplete=True)

    r = requests.get(url)

    # Throw an error if the request was not successful
    if not r.ok:
        panic("FIle at "+url+" could not be downloaded")

    fs.write_file(path, r.content)

    # Complete log
    complete_done("done ({:,} bytes)".format(len(r.content)))

def build_dataset() -> None:
    """
    Downloads and processes all necessary datasets to build a cumulative dataset
    for this project.
    """

    ok = user_warn("In order to build the dataset, about 220 MB of data \
will need to be downloaded from the internet.")

    if not ok:
        panic("Operation cancelled.")

    # DOWNLOAD NECESSARY FILES

    print(colored("Downloading files...", "cyan"))

    # Some of these files are fairly large. If they exist, don't redownload them.
    _download_file(URL_GAZETTER, "_data/tmp/gazetter.zip", overwrite=False)
    _download_file(URL_CENSUS, "_data/tmp/census.csv", overwrite=False)
    _download_file(URL_VACCINATION, "_data/tmp/vacc.csv", overwrite=False)
    _download_file(URL_COVID, "_data/tmp/covid.csv", overwrite=False)
    _download_file(URL_GDP, "_data/tmp/gdp.json", overwrite=False)

    # UNZIP GAZETTER DATABASE

    print(colored("Extracting...", "cyan"))
    with zipfile.ZipFile("_data/tmp/gazetter.zip", 'r') as zip_ref:
        archive_members = zip_ref.filelist

        if not archive_members:
            panic("Zip file unexpectedly empty")

        zip_ref.extract(archive_members[0], "_data/tmp")

    # BUILD US COUNTY DATABASE FROM CENSUS DATA

    info(colored("Building dataset: Compiling county info", "cyan"), incomplete=True)

    counties = CountyDataset()

    with open("_data/tmp/census.csv") as fl:
        reader = csv.reader(fl)

        next(reader)  # Skip header

        for row in reader:
            county_name = row[6]
            state_name = row[5]
            pop_2020 = int(row[-1])

            county = CountyData(
                name=county_name,
                state_abbv=state_name_to_abbv(state_name),
                population=pop_2020,
                # Assign these values to defaults of -1; we will write them later.
                land_area=-1,
                gdp=-1,
                vacc_rate=-1.0,
                geo_lat=-1.0,
                geo_long=-1.0,
                cases=[],
                deaths=[]
            )

            # The csv file records cumulative state data
            # as a county with the name as its state. Ignore these
            if county_name == state_name:
                continue

            counties.add_county(county, ignore_duplicates=True)
    
    complete_done()

    # ADD DATA FROM GAZETTER

    with open("_data/tmp/2020_Gaz_counties_national.txt") as fl:
        reader = csv.reader(fl, delimiter="\t")

        next(reader)

        for row in reader:
            county_name = row[3]
            state_abbv = row[0]

            if county := counties.get_county(county_name, state_abbv):
                county.land_area = float(row[6])
                county.geo_lat = float(row[8])
                county.geo_long = float(row[9])

    # ADD GDP DATA

    rows = json.loads(
        fs.read_file("_data/tmp/gdp.json")
    )["BEAAPI"]["Results"]["Data"]

    for row in rows:
        # Will be something like "Baldwin, AL"
        geo_string = row["GeoName"]

        # Sometimes the geo string has an astrisk at the end? Remove it
        geo_string = geo_string.rstrip("*")

        # Use regex to parse out the state abbv and county name.
        county_name = unwrap(
            re.match(r'.+(?=, [A-Z]{2})', geo_string)).group(0)
        state_abbv = unwrap(re.search(r'[A-Z]{2}$', geo_string)).group(0)

        # Sometimes the county name in this dataset says something like "Wise + Norton".
        # if this is the case, take only the first name.
        if " + " in county_name:
            county_name = county_name.split(" + ")[0]

        # Sometimes the county name in this dataset ends with something like "(Independent City)"
        # if this is the case, remove it
        if "(Independent City)" in county_name:
            county_name = county_name.replace("(Independent City)", "")

        # row["DataValue"] looks something like "4,050,073"
        gdp_string = row["DataValue"].replace(",", "")

        # Sometimes the GDP is listed as (NA)
        # if this is the case, skip this area
        if gdp_string == "(NA)":
            continue
        gdp = int(gdp_string) * 1000  # unit is given in 1000s of dollars

        # search for county instance
        if county := counties.search_for_county(county_name, state_abbv):
            county.gdp = gdp

    # ADD VACCINATION DATA

    with open("_data/tmp/vacc.csv") as fl:
        reader = csv.reader(fl)

        next(reader)

        # Keep track of the date listed on each row; if it starts going back in time, stop.
        curr_date = ""

        for row in reader:
            county_name = row[3]
            state_abbv = row[4]
            vacc_percent = float(row[5])

            # if the current date has not been assigned, assign it
            if not curr_date:
                curr_date = row[0]
            else:
                # otherwise, if the date has changed, we've gone too far; stop
                if row[0] != curr_date:
                    break

            if county := counties.get_county(county_name, state_abbv):
                county.vacc_rate = vacc_percent / 100

    # ADD COVID DATA

    print(colored("Building dataset: Compiling COVID data... (this may take a minute)", "cyan"))

    # first, keep track of the list of dates in the covid dataset.
    # assign each day a number consecutively, starting from 0 being the oldest. we call this a
    # "date index".
    #
    # a date string is stored as "yyyy-mm-yy"

    date_num = 0
    date_indexes: dict[str, int] = {}  # maps date strings to date numbers
    dates: list[str] = [] # the nth item in this list is the date corresponding to the date index n.

    last_date_string = ""

    # compile list of dates
    with open("_data/tmp/covid.csv") as fl:
        reader = csv.reader(fl)

        next(reader)  # skip header

        for row in reader:
            date_string = row[0]

            # assign if stil unitialized
            if last_date_string == "":
                last_date_string = date_string
                dates.append(date_string)

            # check if date has changed
            if date_string != last_date_string:
                date_num += 1
                dates.append(date_string)

            last_date_string = date_string

            # add to the table
            date_indexes[date_string] = date_num

    # Next, populate each county with a list of zeroes for cases/deaths on each day
    # the index of this list represents the data point at that day index.
    #
    # For instance, let the date index 0 represents 2020-01-21, the number 1 represents 2020-01-22,
    # and so on.
    # By saying we have, say, 12 deaths on 1/21/2021, the 0th element of county.deaths would be 12.
    for county in counties.iter():
        county.cases = [0 for _ in range(date_num+1)]
        county.deaths = [0 for _ in range(date_num+1)]

    # The county names in this dataset do not exactly match the ones
    # in our existing dataset. Searching the dataset for a close match
    # is also an expensive operation. Once we do find a matching county, store it
    # so we don't have to look for it again.
    #
    # The key of this dictionary is a (name, abbrevation) tuple
    # and the output is the corresponding CountyData instance
    # in the dataset.
    _county_cache: dict[tuple[str, str], CountyData] = {}

    # take an estminate of no. of rows in the file based on number of newline characters.
    num_rows = fs.read_file("_data/tmp/covid.csv").count("\n")

    # count processed rows
    rows_done = 0

    # time of last status message
    _time_last_status = time.time()

    # read the file again, but this time aggregate data
    with open("_data/tmp/covid.csv") as fl:
        reader = csv.reader(fl)

        next(reader)  # skip header

        for row in reader:
            date_string = row[0]
            county_name = row[1]

            # if county_name is "unknown", don't even bother
            if county_name == "Unknown":
                continue

            state_name = row[2]
            state_abbv = state_name_to_abbv(state_name)

            # get date index from date string
            date_idx = date_indexes[date_string]

            # search for county using county_name or use cache
            county: Union[CountyData, None] = None

            if (county_name, state_abbv) in _county_cache:
                county = _county_cache[(county_name, state_abbv)]
            else:
                # if we haven't encountered it yet (i.e. not in cache)
                # search for it using the dataset search method.
                county = counties.search_for_county(county_name, state_abbv)

                # cache the search result
                if county:
                    _county_cache[(county_name, state_abbv)] = county

            if county:
                # for cases and deaths
                # if the value is invalid for whatever reason, just use the
                # previous data point.
                try:
                    cases = int(row[4])
                except ValueError:
                    cases = county.cases[date_idx - 1]
                except IndexError:
                    cases = county.cases[date_idx - 1]

                try:
                    deaths = int(row[5])
                except ValueError:
                    deaths = county.deaths[date_idx - 1]
                except IndexError:
                    deaths = county.deaths[date_idx - 1]

                county.cases[date_idx] = cases
                county.deaths[date_idx] = deaths
            else:
                # warn(
                #     "unknown county "+county_name +
                #     ", "+state_name_to_abbv(state_name)
                # )

                pass

            rows_done += 1

            # every 2 seconds, print out a status message
            if time.time() - _time_last_status >= 2:
                info("COVID-19 Data Aggregation: {}".format(
                    colored(
                        "{0:.2f}%".format(rows_done/num_rows*100),
                        "cyan"
                    )
                ))

                _time_last_status = time.time()

    # PRUNE DATASET

    print(colored("Cleaning database...", "cyan"))

    # remove any dataset items whos data contains the default number "-1"
    # this means that this dataset item is incomplete and should not be included.

    to_remove: list[CountyData] = []

    for county in counties.iter():
        should_remove = False
        reason: Union[str, None] = None

        # Find "-1" values and mark for removal, citing a reason
        if county.gdp == -1:
            should_remove = True
            reason = "GDP"
        elif county.land_area == -1:
            should_remove = True
            reason = "Land Area"
        elif county.vacc_rate == -1:
            should_remove = True
            reason = "Vaccination"
        
        # Check that covid data is present
        if all([c == 0 for c in county.cases]):
            should_remove = True
            reason = "COVID-19 Case"

        # If anything is missing, mark for removal, and print message.
        if should_remove:
            warn("Incomplete data for county {} ({} data not found), removing...".format(
                colored(county.name + ", " + county.state_abbv, "magenta"),
                reason
            ))

            to_remove.append(county)
            

    # Remove flagged counties
    for county in to_remove:
        counties.remove_county(county)

    # SAVE DATASET

    print(colored("Saving dataset... ({:,} items)".format(
        counties.get_length()), "cyan"))

    counties.save_dataset("_data/dataset.json", dates)

    # CLEAN UP

    print(colored("Cleaning up...", "cyan"))

    if "--dont-cache" in sys.argv:
        fs.remove_path("_data/tmp")
