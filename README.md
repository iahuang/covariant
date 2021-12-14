# COVariant - Investigating Risk Factors for Regional COVID-19 Severity
*This project was originally written for CSC110Y1 at the University of Toronto*

COVariant is a machine learning and data analysis project that tries to predict the overall
impact of the COVID-19 pandemic over time for some locality in the United States given only information
such as its geographic location, population, and vaccination rate. These prediction data are
quantified in terms of their "curve graphs", plots measuring total cases with respect to time.
![Screenshot](https://github.com/iahuang/csc110-term-project/raw/main/github_assets/cover.png)
When the input factors are set for existing US counties and then factors such as vaccination rate
are adjusted, we are able to see how a machine learning model predicts they would have changed
the current outcome of COVID-19. These data not only shows us what factors are likely most
influential in governing the spread of COVID-19 and thus allows us to better implement safety
measures and prepare for future pandemics.

## Usage

Check `requirements.txt` for an up-to-date list of required PyPI packages to run this project.
Note that Tensorflow may have special installation requirements for your system. Training this
model may take anywhere from 2-15 minutes depending on the system used. GPU-enabled training
is supported but not necessary for this project.

```
python main.py
```

### Command Line Arguments
- `--skip-check`
Skip running preliminary checks to see if the runtime environment is properly configured.
- `-y`
Automatically choose "yes" for any confirmation prompts.
- `--dont-cache`
Remove temporary files related to building the dataset after the dataset has finished building. These files will need to be redownloaded next time.
- `--quiet`
Suppress `[LOG]` output.
- `--re-train`
Retrain machine learning model.

## Notice on External Data

This project utilizes public databases from the CDC, New York Times, and other government
organizations. Check the References section in our formal project report below for more information
on the respective copyright and usage information for these datasets.

If for whatever reason, one or more of the dataset URLs no longer works, please create an Issue
on this repository, and a pre-compiled dataset can be provided.

## Results

### Training Observations and Setup
For these results, the machine learning model was trained on 95% of the dataset (about 1.9 million data points), with the remaining 5% used to verify testing/validation loss. The model was trained for one epoch; we found that with subsequent training iterations, the testing loss increased substantially. In total, it took only about three minutes to train the dataset on a laptop with a NVIDIA GTX 1660 Ti. At the end of training, the model had a training loss of `0.002` and a validation loss of `0.018`.

### Example Case
Take, as an example, the county of Providence, Rhode Island. We can see that the model, in general, does a good job at predicting the impact of COVID-19 over time.
![Screenshot](https://github.com/iahuang/csc110-term-project/raw/main/github_assets/v1.png)
As we hypothetically increase the vaccination rate from 66% to 75%, we that both the overall case and death curves drop noticeably
![Screenshot](https://github.com/iahuang/csc110-term-project/raw/main/github_assets/v2.png)
At the vaccination rate approaches 100%, we see diminishing returns in terms of predicted infection; however the over all death toll is predicted to be significantly diminished
![Screenshot](https://github.com/iahuang/csc110-term-project/raw/main/github_assets/v3.png)
These predicted data are consistent with [CDC findings on the more recent Delta Variant](https://www.cdc.gov/coronavirus/2019-ncov/variants/delta-variant.html), in which evidence shows that the Delta variant is infectious, even among those who are vaccinated; however, serious health effects seem to be largely mitigated in most individuals.

## CSC110

This project was originally a Semester Final Project for [CSC110](https://artsci.calendar.utoronto.ca/course/csc110y1)
at the University of Toronto St. George. Our original project report can be found in the `github_assets`
folder or downloaded [here](https://github.com/iahuang/csc110-term-project/raw/main/github_assets/project_report.pdf)