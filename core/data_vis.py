"""
Data Visualization

Module Description
==================
A module for machine learning data visualization

Copyright Information
===============================
This file is Copyright (c) 2021 Deon Chan, Ian Huang, Emily Wan, Angela Zhuo.
"""

from core.county import CountyData, CountyDataset, abbv_to_state_name
from core.ml import COVIDGraphModel, DatasetWrapper
from dataclasses import dataclass
from typing import Callable, Literal, Union, Any, cast
import tkinter as tk
from tkinter import Button, font
from tkinter import Label, Frame, Misc
from PIL import ImageTk, Image
from core.map_state_colors import STATE_COLOR_MAP
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import ctypes

# Attempts to resolve an issue with the Tkinter window blurry
# on some displays.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

class _Globals:
    _current_tk_root_instance: Union[tk.Tk, None] = None


# Functions for setting a global reference to the
# Tkinter root window
def set_current_root(root: tk.Tk) -> None:
    """Set a reference to the root"""
    _Globals._current_tk_root_instance = root


def get_current_root() -> tk.Tk:
    """Get the current root instance; throw an error if has not been assigned."""
    if not _Globals._current_tk_root_instance:
        raise ValueError("Root has not been assigned")

    return _Globals._current_tk_root_instance


class _FontLibrary:
    """
    A simple class for keeping track of fonts across systems.
    """
    _available_fonts: list[str]
    _font_set: set[str]

    # Named font families; potentially system dependent
    sans_serif: str
    monospace: str

    # Default font size
    size_normal: int
    size_large: int
    size_header: int
    size_small: int

    def __init__(self):
        self._available_fonts = []
        self._font_set = set()

        self.size_normal = 12
        self.size_large = 16
        self.size_header = 20
        self.size_small = 10

    def _choose_font(self, fonts: list[str]) -> str:
        """
        Given a list of fonts, choose the first one in the list that is available to use.
        For instance, given _choose_font(["Papyrus", "Comic Sans MS"]), if "Papyrus" is not
        available, choose "Comic Sans"; if neither are available, return TkDefaultFont
        """
        for f in fonts:
            if f in self._font_set:
                return f

        return "TkDefaultFont"

    def get_fonts(self) -> list[str]:
        return [i for i in self._available_fonts]

    def init(self) -> None:
        """This method can only be called after the root window is instantiated"""
        self._available_fonts = list(font.families())
        self._font_set = set(self._available_fonts)

        # Set font families
        self.sans_serif = self._choose_font(
            ["Segoe UI", "San Francisco", "Calibri", "Arial"])
        self.monospace = self._choose_font(
            ["Consolas", "Courier New", "Terminal"])

        # Set default font to sans-serif
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family=self.sans_serif)


@dataclass
class _ColorLibrary:
    black: str = "#000000"
    white: str = "#ffffff"

    bg_1: str = "#ffffff"
    bg_2: str = "#ECEFF1"
    bg_3: str = "#CFD8DC"

    text_1: str = "#263238"
    text_2: str = "#78909C"

    orange: str = "##FF5722"
    blue: str = "#03A9F4"
    teal: str = "#26A69A"
    red: str = "#F44336"


app_fonts = _FontLibrary()
app_colors = _ColorLibrary()


class AppWidgets:
    """A class with methods for constructing widgets according to the app theme"""

    def __init__(self):
        pass

    def text_label(
        self,
        parent: Misc,
        text: str = "",
        font_family: str = None,
        font_size: int = None,
        color: str = app_colors.text_1,
        justify: Literal["center", "right", "left"] = "center"
    ) -> Label:
        if not font_family:
            font_family = app_fonts.sans_serif

        if not font_size:
            font_size = app_fonts.size_normal

        return Label(
            parent,
            text=text,
            font=(font_family, font_size),
            bg=app_colors.bg_1,
            fg=color,
            justify=justify
        )

    def text_button(
        self,
        parent: Misc,
        text: str,
        onclick: Callable,
        font_family: str = None,
        font_size: int = None,
        color: str = app_colors.text_1
    ) -> Button:
        if not font_family:
            font_family = app_fonts.sans_serif

        if not font_size:
            font_size = app_fonts.size_normal

        return Button(
            parent,
            text=text,
            font=(font_family, font_size),
            bg=app_colors.bg_1,
            fg=color,
            cursor="hand2",
            command=onclick
        )

    def image_label(
        self,
        parent: Misc,
        image: ImageTk.PhotoImage
    ) -> Label:
        return Label(parent, image=image, bg=app_colors.bg_1)


app_widgets = AppWidgets()


def _round_size(size: tuple[float, float]) -> tuple[int, int]:
    """Round the size tuple to the nearest integers"""
    return (round(size[0]), round(size[1]))


def _load_image(
    path: str,
    size: tuple[int, int] = None,
    width: int = None,
    height: int = None
) -> ImageTk.PhotoImage:
    """Load an image as a Tk image object"""
    p_img = Image.open(path)
    if size:
        p_img = p_img.resize(size)
    elif width:
        factor = width / p_img.size[0]
        new_size = _round_size((width, p_img.size[1] * factor))
        p_img = p_img.resize(new_size)
    elif height:
        factor = height / p_img.size[1]
        new_size = _round_size((p_img.size[0] * factor, height))
        p_img = p_img.resize(new_size)

    return ImageTk.PhotoImage(p_img)


class ImageCache:
    """
    In Tkinter, if an image object is garbage collected, its corresponding
    picture on screen will disappear. Use this class to ensure that at least one
    reference is kept while necessary.
    """
    _images: list[ImageTk.PhotoImage]

    def __init__(self):
        self._images = []

    def add(self, img: ImageTk.PhotoImage) -> None:
        self._images.append(img)

    def clear(self) -> None:
        self._images.clear()


class CountyDataView(Frame):
    county_data_original: CountyData
    model: COVIDGraphModel

    # UI
    graph_img: ImageTk.PhotoImage
    graph: Label
    model_input_info: Label

    # Modifiable values
    data_population: int
    data_gdp: int
    data_vacc_rate: float
    data_land_area: float

    # text entry widgets
    entry_pop: tk.Entry
    entry_gdp: tk.Entry
    entry_vacc: tk.Entry
    entry_area: tk.Entry

    # callback
    on_back: Callable

    def __init__(self, parent: Misc, county_data: CountyData, model: COVIDGraphModel, on_back: Callable):
        super().__init__(parent, bg=app_colors.bg_1)
        self.county_data_original = county_data

        self.on_back = on_back

        # Load initial values
        self.data_population = self.county_data_original.population
        self.data_gdp = self.county_data_original.gdp
        self.data_vacc_rate = self.county_data_original.vacc_rate
        self.data_land_area = self.county_data_original.land_area

        self.model = model

        # Build initial graph
        self.graph_img = self._graph()
        self.graph = app_widgets.image_label(
            self,
            self.graph_img
        )
        self.graph.pack(side="left", anchor="nw")

        # County Info
        small_header = app_widgets.text_label(
            self,
            "Model Inputs",
            font_size=app_fonts.size_large
        )
        small_header.pack(side="top")

        self.model_input_info = app_widgets.text_label(
            self,
            self.generate_model_input_info(),
            color=app_colors.text_2,
            justify="left"
        )
        self.model_input_info.pack(side="top", anchor="nw")

        # Controls
        instructions = app_widgets.text_label(
            self,
            "The following county data values can be adjusted to see how the model predicts they \
would change the overall impact of COVID-19 over time.",
            font_size=app_fonts.size_small,
            justify="left"
        )
        instructions.config(wraplength=550,)
        instructions.pack(side="top", anchor="nw")

        # Labels and input fields

        # wrap input fields in their own frame
        frame = Frame(
            self,
            bg=app_colors.bg_1
        )
        # Create input field labels
        app_widgets.text_label(
            frame, text="Population", justify="left"
        ).grid(row=0, sticky="w")
        app_widgets.text_label(
            frame, text="GDP", justify="left"
        ).grid(row=1, sticky="w")
        app_widgets.text_label(
            frame, text="Vaccination Rate", justify="left"
        ).grid(row=2, sticky="w")
        app_widgets.text_label(
            frame, text="Land Area", justify="left"
        ).grid(row=3, sticky="w")

        # Create input fields and add default values
        self.entry_pop = tk.Entry(frame, bg=app_colors.bg_1)
        self.entry_pop.insert(tk.END, str(
            self.county_data_original.population))

        self.entry_gdp = tk.Entry(frame, bg=app_colors.bg_2)
        self.entry_gdp.insert(tk.END, str(self.county_data_original.gdp))

        self.entry_vacc = tk.Entry(frame, bg=app_colors.bg_2)
        self.entry_vacc.insert(tk.END, "{:.2f}".format(
            self.county_data_original.vacc_rate))

        self.entry_area = tk.Entry(frame, bg=app_colors.bg_2)
        self.entry_area.insert(tk.END, "{:.2f}".format(
            self.county_data_original.land_area))

        # Align input fields
        self.entry_pop.grid(row=0, column=1)
        self.entry_gdp.grid(row=1, column=1)
        self.entry_vacc.grid(row=2, column=1)
        self.entry_area.grid(row=3, column=1)

        # add buttons
        button = app_widgets.text_button(
            frame,
            "Rerun Model",
            onclick=self.on_click_rerun
        ).grid(row=4, column=0, padx=5)
        button2 = app_widgets.text_button(
            frame,
            "Reset Values",
            onclick=self.on_click_reset
        ).grid(row=4, column=1, padx=5)
        button3 = app_widgets.text_button(
            frame,
            "Back to Map",
            onclick=self.on_back
        ).grid(row=4, column=2, padx=5)

        # Add to scene
        frame.pack(side="top", anchor="nw", expand=True, pady=20)

    def on_click_rerun(self) -> None:
        # if any of the fields are invalid, this variable should be set to "False"
        inputs_ok = True

        try:
            self.data_population = int(self.entry_pop.get())
        except ValueError:
            inputs_ok = False

        try:
            self.data_gdp = int(self.entry_gdp.get())
        except ValueError:
            inputs_ok = False

        try:
            self.data_vacc_rate = float(self.entry_vacc.get())
        except ValueError:
            inputs_ok = False

        try:
            self.data_land_area = float(self.entry_area.get())
        except ValueError:
            inputs_ok = False

        if inputs_ok:
            # Clamp inputs and outputs to appropriate values
            self.data_population = max(self.data_population, 0)
            self.data_gdp = max(self.data_gdp, 0)
            self.data_vacc_rate = max(min(self.data_vacc_rate, 1), 0)
            self.data_land_area = max(self.data_land_area, 0)

            self.model_input_info["text"] = self.generate_model_input_info()
            self.rebuild_graphs()

    def on_click_reset(self) -> None:
        self.data_population = self.county_data_original.population
        self.data_gdp = self.county_data_original.gdp
        self.data_vacc_rate = self.county_data_original.vacc_rate
        self.data_land_area = self.county_data_original.land_area

        self.entry_pop.delete(0, "end")
        self.entry_gdp.delete(0, "end")
        self.entry_vacc.delete(0, "end")
        self.entry_area.delete(0, "end")
        self.entry_pop.insert(tk.END, str(
            self.county_data_original.population))
        self.entry_gdp.insert(tk.END, str(self.county_data_original.gdp))
        self.entry_vacc.insert(tk.END, "{:.2f}".format(
            self.county_data_original.vacc_rate))
        self.entry_area.insert(tk.END, "{:.2f}".format(
            self.county_data_original.land_area))

    def generate_model_input_info(self) -> str:
        return "\n".join([
            "STATE: {}".format(self.county_data_original.state_abbv),
            "POPULATION: {:,}".format(self.data_population),
            "LAND AREA: {:,} sq. miles".format(round(self.data_land_area)),
            "POP. DENSITY: {:,.2f} people per sq. mile".format(
                self.data_population/self.county_data_original.land_area
            ),
            "GDP: ${:,} USD".format(self.data_gdp),
            "VACCINATION RATE: {:.1f}%".format(self.data_vacc_rate*100),
            "COUNTY LATITUDE: {:.4f}".format(
                self.county_data_original.geo_lat),
            "COUNTY LONGITUDE: {:.4f}".format(
                self.county_data_original.geo_long),
        ])

    def _graph(self) -> ImageTk.PhotoImage:
        fig, axes = plt.subplots(2, 1, sharex="col")
        cases_subplot: Axes = axes[0]
        deaths_subplot: Axes = axes[1]

        # x-axis (time) is days from 0..i where
        # i is the greatest date index in the dataset
        time_plot = list(range(len(self.county_data_original.cases)))

        prediction = self.model.predict(CountyData(
            name=self.county_data_original.name,
            state_abbv=self.county_data_original.state_abbv,
            population=self.data_population,
            land_area=self.data_land_area,
            gdp=self.data_gdp,
            vacc_rate=self.data_vacc_rate,
            geo_lat=self.county_data_original.geo_lat,
            geo_long=self.county_data_original.geo_long,
            cases=self.county_data_original.cases,
            deaths=self.county_data_original.deaths
        ))

        # Generate plot data

        actual_cases_plot: list[float] = []
        actual_deaths_plot: list[float] = []
        pred_cases_plot: list[float] = []
        pred_deaths_plot: list[float] = []

        date_idx = 0
        for point in prediction:
            pred_cases_plot.append(point.predicted_cases)
            pred_deaths_plot.append(point.predicted_deaths)
            actual_cases_plot.append(self.county_data_original.cases[date_idx])
            actual_deaths_plot.append(
                self.county_data_original.deaths[date_idx])

            date_idx += 1

        # Format graphs

        dataset = self.model.get_dataset_wrapper().get_county_data()
        day_zero = dataset.dates[0]

        cases_subplot.set_ylabel("Total Cases")
        deaths_subplot.set_ylabel("Total Deaths")
        deaths_subplot.set_xlabel("Days since "+day_zero)

        cases_subplot.plot(
            time_plot,
            pred_cases_plot,
            label="Predicted"
        )
        cases_subplot.plot(
            time_plot,
            actual_cases_plot,
            label="Historical"
        )

        deaths_subplot.plot(
            time_plot,
            pred_deaths_plot,
            label="Predicted"
        )
        deaths_subplot.plot(
            time_plot,
            actual_deaths_plot,
            label="Historical"
        )

        cases_subplot.legend()
        deaths_subplot.legend()

        fig.suptitle("COVID-19 Historical Data and Model Predictions based on \
County Statistics for {}, {}".format(
            self.county_data_original.name,
            self.county_data_original.state_abbv
        ))

        # Export
        PATH = "_data/tmp/graph.png"
        fig.set_size_inches(10, 10)
        plt.savefig(PATH, dpi=80)
        plt.close()

        return _load_image(PATH, height=720)

    def rebuild_graphs(self) -> None:
        self.graph_img = self._graph()
        self.graph["image"] = self.graph_img


class CountySelect(Frame):
    listbox: tk.Listbox
    dataset: CountyDataset
    state_abbv: str
    counties: list[CountyData]
    list_vars: tk.StringVar
    label_county_info: tk.Label
    selected_county: Union[CountyData, None]

    on_select_county: Callable

    def __init__(
        self,
        parent: Misc,
        county_dataset: CountyDataset,
        state_abbv: str,
        onselect: Callable
    ):
        super().__init__(parent, bg=app_colors.bg_1)
        self.dataset = county_dataset
        self.state_abbv = state_abbv
        self.selected_county = None
        self.on_select_county = onselect

        # find corresponding counties for this state
        self.counties = []
        for c in county_dataset.get_counties():
            if c.state_abbv == state_abbv:
                self.counties.append(c)

        # make title
        title = app_widgets.text_label(
            self,
            "Counties in "+abbv_to_state_name(state_abbv),
            font_size=app_fonts.size_header
        )
        title.pack(side="top")

        # make subtitle
        subtitle = app_widgets.text_label(
            self,
            "({:,} counties listed)".format(len(self.counties)),
            color=app_colors.text_2)
        subtitle.pack(side="top")

        # make listbox
        self.listbox = tk.Listbox(self)
        # bind event handler
        self.listbox.bind("<<ListboxSelect>>", self.on_selection_change)

        # populate listbox
        i = 1
        for county in self.counties:
            self.listbox.insert(i, county.name)
            i += 1

        self.listbox.pack(side="top", anchor="nw", padx=20, fill="both")

        # make county info label
        self.label_county_info = app_widgets.text_label(
            self,
            "(No county selected)",
            justify="left",
        )
        self.label_county_info.config(padx=20)
        self.label_county_info.pack(side="top", anchor="nw")

        # make button
        button = app_widgets.text_button(
            self,
            "View Data",
            onclick=self.on_select_button_click
        )

        button.pack(side="top", anchor="nw", padx=20)

    def on_selection_change(self, event: tk.Event) -> None:
        self.selected_county = self.counties[cast(
            tk.Listbox, event.widget).curselection()[0]]
        self.label_county_info["text"] = "Population: {:,}\nCurrent Vaccination Rate: {}%".format(
            self.selected_county.population,
            round(self.selected_county.vacc_rate * 100)
        )

    def on_select_button_click(self) -> None:
        if self.selected_county:
            self.on_select_county(self.selected_county)


class StateSelect(Frame):
    img_us_map: ImageTk.PhotoImage
    color_coded_map: Image.Image
    map_label: Label

    selected: Label
    _state_selected: Union[str, None]

    onselect: Callable

    def __init__(self, parent: Misc, onselect: Callable):
        super().__init__(parent, bg=app_colors.bg_1)

        title = app_widgets.text_label(
            self,
            text="Select a US state to view data",
            font_size=app_fonts.size_header
        )

        self.selected = app_widgets.text_label(
            self,
            text="(None Selected)",
            color=app_colors.text_2
        )

        title.pack(side="top")
        self.selected.pack(side="top")

        self.img_us_map = _load_image("resources/map.png", width=960)
        self.map_label = app_widgets.image_label(self, image=self.img_us_map)
        self.map_label.pack(side="top", anchor="nw")
        # Bind events
        self.map_label.bind("<Motion>", self.on_hover)
        self.map_label.bind("<Button-1>", self.on_click)

        # Load color-state mapping and resize to the size of the on-screen map
        self.color_coded_map = Image.open("resources/map_color.png")
        factor = 960 / self.color_coded_map.size[0]
        self.color_coded_map = self.color_coded_map.resize(
            _round_size((960, self.color_coded_map.size[1] * factor))
        )

        self.onselect = onselect
        self._state_selected = None

    def on_hover(self, event: tk.Event) -> None:
        self.selected["text"] = "(None Selected)"
        self._state_selected = None
        get_current_root().config(cursor="arrow")
        try:
            red_val = self.color_coded_map.getpixel((event.x, event.y))[0]
            if red_val != 0:
                state_abbv = STATE_COLOR_MAP[(red_val, 0, 0)]

                self.selected["text"] = abbv_to_state_name(
                    state_abbv)  # update text onscreen
                self._state_selected = state_abbv  # update internal state

                get_current_root().config(cursor="hand2")  # show pointer cursor

        except KeyError:
            pass
        except IndexError:
            pass

    def on_click(self, _: tk.Event) -> None:
        # reset cursor before moving to the next menu
        get_current_root().config(cursor="arrow")
        if self._state_selected:
            self.onselect(self._state_selected)


class ModelVisualizer:
    model: COVIDGraphModel
    dataset_wrap: DatasetWrapper
    county_dataset: CountyDataset
    max_date_index: int

    _root: tk.Tk
    frame: tk.Frame

    def __init__(self, model: COVIDGraphModel):
        self.model = model
        self.dataset_wrap = model.get_dataset_wrapper()
        self.county_dataset = self.dataset_wrap.get_county_data()
        self.max_date_index = len(self.county_dataset.dates) - 1

        self._root = tk.Tk()
        self._root.configure(bg="red")

        set_current_root(self._root)

        app_fonts.init()

        self._root.geometry("1280x720")
        self._root.resizable(False, False)

        self.frame = cast(Any, None)
        self.go_to_state_map()


    def go_to_state_map(self) -> None:
        if self.frame:
            self.frame.destroy()

        self.frame = StateSelect(self._root, onselect=self.on_select_state)

        self.frame.pack(
            side="top",
            fill="both",
            expand=True
        )

    def on_select_state(self, state_abbv: str) -> None:
        self.frame.destroy()
        self.frame = CountySelect(
            self._root,
            self.county_dataset,
            state_abbv,
            onselect=self.on_select_county
        )
        self.frame.pack(
            side="top",
            fill="both",
            expand=True
        )

    def on_select_county(self, county: CountyData) -> None:
        self.frame.destroy()
        self.frame = CountyDataView(
            self._root,
            county,
            self.model,
            on_back=self.go_to_state_map
        )
        self.frame.pack(
            side="top",
            fill="both",
            expand=True
        )

    def run(self) -> None:
        self._root.mainloop()
