"""Copyright Information
===============================
This file is Copyright (c) 2021 Deon Chan, Ian Huang, Emily Wan, Angela Zhuo.
"""

from typing import Any, Callable, cast
import tensorflow as tf
import tensorflow.keras as keras
from core.county import CountyData, CountyDataset
from core.logger import complete_done, info
import numpy as np
import math
from dataclasses import dataclass

class DatasetNorm:
    """
    Stores information about dataset maximum and minimum values
    for data such as land area, etc. This way, the data can be
    normalized when passed to the neural network.
    """
    _pop_max: int
    _land_area_max: float
    _gdp_max: int
    _pop_density_max: float
    # Geographic data
    _lat_max: float
    _lat_min: float
    _long_max: float
    _long_min: float

    # For case data
    _max_date: int
    _covid_deaths_max: int
    _covid_cases_max: int
    _covid_deaths_rate_max: float
    _covid_cases_rate_max: float

    # state data
    _states: list[str]

    def __init__(self):
        self._pop_max = 0
        self._land_area_max = 0
        self._gdp_max = 0
        self._pop_density_max = 0.0
        self._lat_max = 0.0
        self._lat_min = 0.0
        self._long_max = 0.0
        self._long_min = 0.0
        self._max_date = 0
        self._covid_deaths_max = 0
        self._covid_cases_max = 0
        self._covid_deaths_rate_max = 0.0
        self._covid_cases_rate_max = 0.0
        self._states = []

    def add_county(self, county: CountyData) -> None:
        """
        Updates min/max values using the provided county, if needed.
        """

        pop_density = county.population / county.land_area

        # geographic data
        self._pop_max = max(county.population, self._pop_max)
        self._land_area_max = max(county.land_area, self._land_area_max)
        self._gdp_max = max(county.gdp, self._gdp_max)
        self._lat_max = max(county.geo_lat, self._lat_max)
        self._lat_min = min(county.geo_long, self._lat_min)
        self._long_max = max(county.geo_long, self._long_max)
        self._long_min = min(county.geo_long, self._long_min)
        self._pop_density_max = max(pop_density, self._pop_density_max)

        # Case/Death data
        self._max_date = max(self._max_date, len(county.cases) - 1)
        self._covid_cases_max = max(
            self._covid_cases_max,
            max(county.cases)
        )
        self._covid_deaths_max = max(
            self._covid_deaths_max,
            max(county.deaths)
        )

        """COVID RATE DATA IS NO LONGER USED; THIS CODE IS COMMENTED FOR PERFORMANCE REASONS"""
        # self._covid_deaths_rate_max = max(
        #     self._covid_deaths_rate_max,
        #     max(county.get_avg_deaths_per_day())
        # )
        # self._covid_cases_rate_max = max(
        #     self._covid_cases_rate_max,
        #     max(county.get_avg_cases_per_day())
        # )

        # This is very inefficient, however, the process_county
        # method is not called many times.
        if county.state_abbv not in self._states:
            self._states.append(county.state_abbv)
        self._states.sort()

    """
    The following methods are functionally similar;
    
    normalize_xyz(xyz) takes a value and compresses it into a 0..1 range.
    un_normalize_xyz(norm_xyz) takes a value from 0..1 and expands it back into its corresponding
    value.
    """

    """
    Some values are normalized in a logarithmic scale,
    since variables such as cases and GDP are centered skewed heavily towards 0
    """
    def _lognorm(self, x: float, max_x: float) -> float:
        return math.log(x + 1) / math.log(max_x)
    
    def _un_lognorm(self, norm_x: float, max_x: float) -> float:
        return math.pow(max_x, norm_x) - 1

    def normalize_deaths(self, deaths: float) -> float:
        return self._lognorm(deaths, self._covid_deaths_max)

    def un_normalize_deaths(self, norm_deaths: float) -> float:
        return self._un_lognorm(norm_deaths, self._covid_deaths_max)

    def normalize_cases(self, cases: float) -> float:
        return self._lognorm(cases, self._covid_cases_max)

    def un_normalize_cases(self, norm_cases: float) -> float:
        return self._un_lognorm(norm_cases, self._covid_cases_max)
    
    def normalize_case_rate(self, case_rate: float) -> float:
        return case_rate / self._covid_cases_rate_max
    
    def un_normalize_case_rate(self, norm_case_rate: float) -> float:
        return norm_case_rate * self._covid_cases_rate_max
    
    def normalize_death_rate(self, death_rate: float) -> float:
        return death_rate / self._covid_deaths_rate_max
    
    def un_normalize_death_rate(self, norm_death_rate: float) -> float:
        return norm_death_rate * self._covid_deaths_rate_max

    def normalize_pop(self, population: int) -> float:
        return self._lognorm(population, self._pop_max)

    def un_normalize_pop(self, norm_population: float) -> int:
        return round(self._un_lognorm(norm_population, self._pop_max))

    def normalize_land_area(self, land_area: float) -> float:
        return self._lognorm(land_area, self._land_area_max)

    def un_normalize_land_area(self, norm_land_area: float) -> float:
        return self._un_lognorm(norm_land_area, self._land_area_max)

    def normalize_gdp(self, gdp: int) -> float:
        return self._lognorm(gdp, self._gdp_max)

    def un_normalize_gdp(self, norm_gdp: float) -> int:
        return round(self._un_lognorm(norm_gdp, self._gdp_max))

    def normalize_pop_density(self, pop_density: float) -> float:
        return self._lognorm(pop_density, self._pop_density_max)

    def un_normalize_pop_density(self, norm_pop_density: float) -> float:
        return self._un_lognorm(norm_pop_density, self._pop_density_max)

    def normalize_lat_long(self, lat: float, long: float) -> tuple[float, float]:
        return (
            (lat - self._lat_min) / (self._lat_max - self._lat_min),
            (long - self._long_min) / (self._lat_max - self._long_min)
        )

    def un_normalize_lat_long(self, norm_lat: float, norm_long: float) -> tuple[float, float]:
        return (
            (norm_lat * (self._lat_max - self._lat_min)) + self._lat_min,
            (norm_long * (self._long_max - self._long_min)) + self._long_min
        )

    def normalize_date(self, date_index: int) -> float:
        return date_index / self._max_date

    def state_as_index(self, state_abbv: str) -> int:
        """Return a state index from its corresponding abbreviation"""
        return self._states.index(state_abbv)

    def index_to_state(self, index: int) -> str:
        """Convert a state index into a state abbrevation"""
        return self._states[index]

    def num_states(self) -> int:
        """Return the number of states in the dataset"""
        return len(self._states)

    def get_max_date_index(self) -> int:
        """Return the maximum allowed date index value"""
        return self._max_date


def norm_from_dataset(dataset: CountyDataset) -> DatasetNorm:
    norm = DatasetNorm()

    for county in dataset.get_counties():
        norm.add_county(county)

    return norm

DatasetType = tuple[np.ndarray, np.ndarray]

class DatasetWrapper:
    _county_data: CountyDataset
    _norm: DatasetNorm
    
    target_value: str

    def __init__(self, county_dataset: CountyDataset, norm: DatasetNorm, target_value: str):
        self._county_data = county_dataset
        self._norm = norm

        self.target_value = target_value

    def create_input_matrix(self, county: CountyData, date_index: int) -> np.ndarray:
        """Return a model input ndarray from county data and a time value"""
        state_vector = [0 for _ in range(self._norm.num_states())]
        state_vector[self._norm.state_as_index(county.state_abbv)] = 1

        population_density = county.population / county.land_area

        return np.array([
            *self._norm.normalize_lat_long(county.geo_lat, county.geo_long),
            *state_vector,
            self._norm.normalize_pop(county.population),
            self._norm.normalize_pop_density(population_density),
            self._norm.normalize_gdp(county.gdp),
            county.vacc_rate,  # this value is already 0-1; it doesn't need normalization
            self._norm.normalize_date(date_index)
        ])
    
    def create_output_matrix(self, county: CountyData, date_index: int) -> np.ndarray:
        """Return a label ndarray for training or testing"""
        return np.array([
            self._norm.normalize_cases(county.cases[date_index]),
            self._norm.normalize_deaths(county.deaths[date_index])
        ])
    
    def get_county_data(self) -> CountyDataset:
        return self._county_data
    
    def make_training_val_sets(self, ratio: float) -> tuple[DatasetType, DatasetType]:
        """Generate training and validation sets from the internal database."""
        train_counties: list[CountyData] = []
        val_counties: list[CountyData] = []

        for i, county in enumerate(self._county_data.get_counties()):
            if i < ratio * self._county_data.get_length():
                train_counties.append(county)
            else:
                val_counties.append(county)
        
        return (
            self._make_dataset_from_counties(train_counties),
            self._make_dataset_from_counties(val_counties)
        )
    
    def make_training_set(self) -> DatasetType:
        """
        Turn the entire database into a training set. Similar to
        make_training_val_sets().
        """
        return self._make_dataset_from_counties(list(self._county_data.get_counties()))
    
    def get_max_date_index(self) -> int:
        return self._norm.get_max_date_index()

    def _make_dataset_from_counties(self, counties: list[CountyData]) -> DatasetType:
        """
        Turns county data into model inputs and labels
        """
        inputs: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for county in counties:
            for date_index in range(self.get_max_date_index() + 1):
                x = self.create_input_matrix(county, date_index)
                y = self.create_output_matrix(county, date_index)

                inputs.append(x)
                labels.append(y)
        
        return (np.array(inputs), np.array(labels))
    
    def get_input_size(self) -> int:
        sample = next(c for c in self._county_data.get_counties())
        return len(self.create_input_matrix(sample, 0))

@dataclass
class DataPoint:
    date_index: int
    predicted_cases: int
    predicted_deaths: int

class COVIDGraphModel:
    """
    A high level interface class for a machine learning model
    """
    
    _dataset_wrap: DatasetWrapper
    _model: keras.Model

    def __init__(self, county_dataset: CountyDataset):
        self._dataset_wrap = DatasetWrapper(
            county_dataset=county_dataset,
            norm=norm_from_dataset(county_dataset),
            target_value="cases"
        )

        input_size = self._dataset_wrap.get_input_size()

        inputs = keras.Input(shape=(input_size,), dtype="float32")
        x = keras.layers.Dense(512, activation="relu")(inputs)
        x = keras.layers.Dense(512, activation="relu")(x)
        x = keras.layers.Dense(512, activation="relu")(x)
        outputs = keras.layers.Dense(2, activation="relu")(x)
        self._model = keras.Model(inputs, outputs)
        info("Compiling model", incomplete=True)
        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.MeanSquaredError()
        )
        complete_done()
            
    def train(self, train_val_ratio: float, epochs: int) -> None:
        info("Building training data (this may use up to 3 GB of RAM)", incomplete=True)
        train, val = self._dataset_wrap.make_training_val_sets(train_val_ratio)
        complete_done("done ({:,} data points)".format(len(train[0])))
        inputs, labels = train
        info("Ready.")
        self._model.fit(
            x=inputs,
            y=labels,
            batch_size=32,
            epochs=epochs,
            validation_data=val,
            shuffle=True
        )
    
    def save(self) -> None:
        self._model.save("_data/model")
    
    def load(self) -> None:
        self._model = cast(Any, keras.models.load_model("_data/model"))
    
    def predict(self, county: CountyData) -> list[DataPoint]:
        """
        Given a CountyData object, return a list of DataPoint objects
        that represents the model's predicted COVID curve with respect to time.
        """
        inputs = []
        for date_index in range(self._dataset_wrap.get_max_date_index() + 1):
            point_x = self._dataset_wrap.create_input_matrix(county, date_index)

            inputs.append(point_x)

        output = self._model.predict(np.array(inputs))

        return [
            DataPoint(
                date_index=i,
                predicted_cases=round(self._dataset_wrap._norm.un_normalize_cases(c[0])),
                predicted_deaths=round(self._dataset_wrap._norm.un_normalize_deaths(c[1]))
            )
            for i, c in enumerate(output)
        ]
    
    def get_dataset_wrapper(self) -> DatasetWrapper:
        return self._dataset_wrap
