# nDrought notebooks

a. stein

Brief description of notebooks and what products they produce. Dates are included for reference of what was most recently worked on. Note that dates are typically only recorded at notebook creation, although I try to note when I come back to something that I haven't in quite a while.

Products are stored in `/pool0/home/steinadi/data/drought/drought_impact/data` with drought specific data being in the `drought_measures` sub-directory. This is what paths refer to throughout the notebook. If you are uncertain where a file is located, ask adi. If adi is uncertain, best of luck.

Updated: 6.8.2022

------------------------------------------

## analysis

Examining data with a specific objective.

### **`agu_plots.ipynb`**
November 2022

Plots for the AGU 2022 Fall Meeting.

### **`all_dm_compare_paired.ipynb`**
8.8.2022

This notebook begins to get into developing the comparison matrix between:
- USDM
- SPI
- SPEI
- EDDI
- PDSI
- Palmer Z Index
- GRACE soil moisture data

By "USDM Date Paired," I mean that while this data will be all categorized into USDM categories for similar measures, their time steps have been synced to USDM, but spatial scales have not been altered.

**Produces**: `*_WA_caf.nc`, `*_drought_counts.pickle`, `*_net.pickle`, `*_WA_nx.pickle`

### **`compare_indicators.ipynb`**

01.12.2022

Provides brief documentation on indicators collected (EDDI, SPEI, SPI, ESI, FDSI, GRACE, LERI, MCDI, Palmer Z Index, PDSI, VEGDRI) regarding what their full names are and where some resources are that describe how they're made. Also compiled is some summary information about the data collect:

| Indicator | Interval | Start | End  | dtype                   |
| --------- | -------- | ----- | ---- | ----------------------- |
| fdsi      | yearly   | 1980  | 2020 | int32                   |
| esi       | monthly  | 2000  | 2019 | `cftime.DatetimeJulian` |
| grace     | weekly   | 2014  | 2021 | `datetime64[ns]`        |
| leri      | monthly  | 2015  | 2021 | object                  |
| mcdi      | monthly  | 1895  | 2021 | `datetime64[ns]`        |
| palmer z  | 5 day    | 1980  | 2021 | `datetime64[ns]`        |
| pdsi      | 5 day    | 1980  | 2021 | `datetime64[ns]`        |
| vegdri    | weekly   | 2019  | 2020 | `datetime64[ns]`        |
| eddi      | 5 day    | 1980  | 2021 | `datetime64[ns]`        |
| spei      | 5 day    | 1980  | 2021 | `datetime64[ns]`        |
| spi       | 5 day    | 1980  | 2021 | `datetime64[ns]`        |

The notebook also includes some summary time series and CDF plots for quick view.

### **`fdsi_and_mtrainer.ipynb`**

12.03.2021

Combines FDSI and Mt. Rainier National Park visitation data in an effort to look for a relationship.

### **`nexus_correlations.ipynb`**
10.11.2022

Paired Pixel Correlation (PPC) between metrics.

### **`nexus_scatters.ipynb`**
9.14.2022

Developed temporal pairing between two metrics, and improved workflow for spatial pairing between two metrics as well. PPC used in `nexus_correlations.ipynb` was first explored here, then continued there for cleanliness.

### **`nonparametric_met_regression.ipynb`**

03.16.2022

This notebook builds off of `park_met_relations.ipynb` where compiling meteorology timeseries for each park was developed and parametric regression (that is mostly non-applicable due to the non-normal nature of the visitation data) was explored. Here I build off of that to use better suited statistical tools for non-normal distributions, aka non-parametric regression and tests.

### **`park_anomaly.ipynb`**

03.08.2022

Develop functions to plot and extract anomalies for park data.

### **`park_met_relations.ipynb`**

03.10.2022

Examine relationships and potential for correlation between park visitation data and meteorological data, (currently precipitation, max air temperature, minimum air temperature).

### **`small_nexus_2015_drought.ipynb`**
11.1.2022

So drawing the entire nexus is too big, even for `pyvis`. Also, doing any computations on a graph that has multiple drought threads (meaning a discontinuous graph) greatly interferes with efforts to measure connectivity. So here I want to try isolating a drought thread to do a smaller scale analysis. For this I'll use the 2015 drought.

Ultimately the attempt was abandoned due to the smaller scale still not working well.

### **`usdm_spi_catfreq.ipynb`**

05.11.2022

Following the other `analysis/usdm_spi_compare_*.ipynb` ... this notebook further complicates spatio-temporal relations by looking at how often each cell is in each category as an aggregation scheme across time (and different time groupings) instead of spatial groupings.

**Produces**: `ndrought_products/paired_USDM_SPI_time_freq.ipynb`

### **`usdm_spi_compare_spatial.ipynb`**

04.27.2022

Building off of `explore/usdm_spi_explore_workflow.ipynb`, this notebook builds upon lessons learned and the workflow developed there to more thoroughly compare the USDM and various interval SPIs.

**Produces**: `spi/WA/spi_usdmcat_WA.nc`, `ndrought_products/paired_USDM_SPI.nc`

### **`usdm_spi_compare_temporal.ipynb`**

05.02.2022

Continuing from `explore/usdm_spi_explore_workflow.ipynb` and `analysis/usdm_spi_compare_spatial.ipynb`, this notebook aims to compare USDM and various SPI intervals temporally, as opposed to spatially.

**Produces**: `ndrought_products/paired_UDSM_SPI_huc4split.nc`, `ndrought_products/paired_USDM_SPI_huc8split.nc`

### **`usdm_spi_investigate_temporal.ipynb`**

05.25.2022

The `analysis/usdm_spi_compare_temporal.ipynb` described rather poor fitting between USDM and SPI temporally ... even when spatial resolution was increased. This is surprising given how SPI should heavily influence a lot of drought metrics, including USDM. In this notebook I'll see if I can tease apart why this mismatch is occuring.

### **`usdm_spi_pixel_compare.ipynb`**

06.02.2022

Instead of aggregating into area fractions, this notebook aims to do a pixel resolution comparison between USDM and SPI.

------------------------------------------

## explore

Delving into data without any particular objective, or looking to develop a workflow but not perform analysis.

### **`animate_drought_network_thread.ipynb`**
8.31.2022

Now that the adjacency dictionary is implemented fine thanks to `drought_network_graph_manipulate.ipynb` and easy to maneuver, let's try to get an animation out of a pulled out thread returned to array.

### **`chord_diagrams.ipynb`**
11.28.2022

Developing some chord diagrams to make plots, exploring with various summary metrics and filtering to best highlight stories in the data.

### **`drought_center_of_mass.ipynb`**
7.7.2022

Perhaps a way to evaluate spatial changes is to calculate a center of mass weighted by higher drought categories having a bigger drag on the center.

### **`drought_declarations.ipynb`**

Date unknown.

Beginning exploration of WA Drought declaration distribution and frequency.

### **`drought_event_comparison.ipynb`**
11.3.2022

Goal: compare evolution of drought between 2 different metrics.

First development of Alignment Fraction (AF), Disagreement Fraction (DF), and Inertia for the Washington 2015 drought between USDM and PDSI.

### **`drought_event_network.ipynb`**
7.28.2022

Okay. So following `explore/drought_tracking.ipynb` I did `quality_control/tset_drought_event_plot.ipynb` and found that while blob identifying and connecting over time works ... the id system is a nightmare and not very smooth to use, especially as I realized during testing that I needed one extra set of parenthesis around a split to make the id's unique. This makes sorting them a further pain and there isn't a great way to trace history despite the id being unique (writing something to then process sorting is really annoying). So, let's try making a nodal network to keep track of it instead in combination with networkx.

### **`drought_event_tracking.ipynb`**
7.8.2022

A useful piece of information would be how big different droughts get because each pixel is impacted by its neighbor. Through blob tracking, I might be able to detect different drought events and their areas over time as they come and go.

### **`drought_gradient.ipynb`**
???
In progress

### **`drought_network_filter_and_mapping.ipynb`**
8.16.2022

This notebook delves into the `DroughtNetwork` class developed in `explore/drought_event_network` and `explore/network_refinement`. Here I look into filtering by area to reduce noise (and disconnecting appropriate events where needed due to filter), adding a few more plot coloring, and converting the DroughtNetwork back out into an image.

### **`drought_network_graph_manipulation.ipynb`**
8.29.2022

Upon wanting to improve filtering and other graph manipulation operations, I'm seeing the adjacency matrix as perhaps not the best method to do this. Here I will explore with adjacency lists instead and being able to include/exclude nodes with (hopefully) greater easy and coding simplicity.

### **`drought_network_overlap.ipynb`**
9.20.2022

Here I want to look into tracing how the same drought evolves differently in different metrics.

Finds overlapping events between two drought networks.

### **`drought_network_refinement.ipynb`**
8.4.2022

This picks up where `explore/drought_event_network.ipynb` left off, presenting a cleaner demonstration of the test case and cleaning up plotting functionality.

### **`drought_network_stats.ipynb`**
8.29.2022

This notebook delves into comparing drought networks with a bit of a delve into graph theory.

### **`drought_network_weighted_nexus.ipynb`**
9.22.2022

So this is a bit of an experimental idea. In exploring the CAF plots, I came to realize that just because the percent of the state in drought might be the same between two measures, doesn't mean it's the same drought. Only considering this aggregated measure looses the diverse topography of the state and the nuances in how drought evolves differently between metrics. This makes the drought network, and examining overlapping events (`explore/drought_network_overlap.ipynb`), rather important when comparing drought evolution and describing similarities ... which can be done with side-by-side animations that were also developed in that notebook, but it would be helpful to have more analytic means of comparing.

This is where the Drought Nexus takes actual shape. I plan to connect the graphs of multiple drought networks via these overlaps. To preserve identification of which node goes to which measure, I'll need to rename the id's something like `dm_{i}`. But then there is one further step I want to take: weighting the graph by area ratios. Each edge will be weighted as a ratio of origin_area to destination_area (if a --> b, then the weight of their connecting edge would be b_area/a_area). This will allow for an understanding of how the drought event is growing (weight > 1), shrinking (weight < 1), or staying the same (weight = 1). As for the weights then between drought measure graphs, which is the numerator and which the denominator will need to be specified ... or an edge could be created each way, following destination_area/origin_area

With the Drought Nexus graph, I can visualize connectivity patterns between all the drought metrics, and see what is commonly agreed upon and what is unique.

### **`explore_animation.ipynb`**

Date unknown.

Explores creating animations.

### **`explore_Google_trends.ipynb`**

Date unknown.

Delves into querying and using data from Google Trends.

### **`explore_licenses.ipynb`**

Date unknown.

This notebook delves into a preliminary analysis of the hunting and fishing data organized in `organize_data/organize_licenses.ipynb`

### **`explore.ipynb`**

Date unknown.

This notebook serves to begin examining datasets, determine their data accessibility, and determine how best to read them into pandas.

### **`usdm_spi_explore_snapshots.ipynb`**

06.02.2022

To get a better understanding of what USDM and SPI looks like across the years, this notebook creates some snapshot plots to look at drought metric evolution with time.

### **`usdm_spi_explore_workflow.ipynb`**

04.18.2022

Explores rescaling USDM & SPI to match each other, creating the UDSM colorbar, how to pair USDM and SPI dates, and categorizing SPI into USDM categories.

**Produces**: `spi/WA/spi_{interval}.nc`

------------------------------------------

## organize_data

Compiling, sorting, and developing structure to data.

### **`clip_cat_dms_graceless.ipynb`**
9.9.2022

Following the work of `organize_data/clip_cat_spei.ipynb`, this notebook looks to clip and categorize the remaining drought measures since the workflow is fairly identical. I want to only use the ones that are weekly at sparsest. These are:
- grace (REMOVED FROM THIS NOTEBOOK)
- palmer z
- pdsi

### **`clip_cat_dms.ipynb`**
7.20.2022

Following the work of `organize_data/clip_cat_spei.ipynb`, this notebook looks to clip and categorize the remaining drought measures since the workflow is fairly identical. I want to only use the ones that are weekly at sparsest. These are:
- grace
- palmer z
- pdsi

**Produces**: clipped & categorized datasets as described

### **`clip_cat_eddi.ipynb`**


### **`clip_cat_spei.ipynb`** 

06.06.2022'

Getting SPEI setup to compare to WA USDM.

**Produces**: `spei/WA/spei_{interval}.nc`

### **`clip_met_data.ipynb`**

03.09.2022

Combine and clip the met data gathered from the THREDDS servers.

**Produces**: 
- `met/precip/clipped_precip_1979.nc`
- `met/precip/clipped_precip_1979_1981.nc`
- `met/precip/wa_or_precip_1979_2022.nc`
- `met/tair_max/wa_or_tair_max_1979_2022.nc`
- `met/tair_min/wa_or_tair_min_1979_2022.nc`

### **`clip_USDM.ipynb`**

04.19.2022

Trims the USDM files down from Global to CONUS and WA separately.

**Produces**: 
- `usdm/CONUS/UDSM_*.nc`
- `usdm/USDM_CONUS_20000104_20220412.nc`
- `usdm/WA/USDM_*.nc`
- `usdm/USDM_WA_20000104_20220412.nc`

### **`format_national_parks.ipynb`**

Date unknown.

This notebooks formats and begins to look at data from National parks with the goal of organizing their information into a netcdf file.

**Produces**: `national_parks_v1.nc`, `national_parks_v2.nc`

### **`format_state_parks.ipynb`**

This notebook picks up after `format_national_parks.ipynb` to format the state park data.

**Produces**: `wa_state_parks.nc`, `or_state_parks.nc`

### **`organize_licenses.ipynb`**

02.03.2022

Processes and organizes hunting and fishing license data.

**Produces**: `DatabaseDrafts/HuntingLicenses_OR/OR_Hunting_Fishing_Licenses_2016_2020.csv`

------------------------------------------

## quality_control

Checking on work done in ways that typically gets more in the weeds and away from analysis itself to verify product accuracy.

### **`dm_quantile_check.ipynb`**
9.7.2022

Here I want to double check that the quantiles used to categorize into USDM categories are sound, or at least as sound as they can be without an objective truth anywho.

### **`grace_data_check.ipynb`**
Week of 8.22.2022

Some of the GRACE data looks suspicious, let's dive into it and check.

### **`open_mfdataset_test`**

06.07.2022 - produced by B. Nijssen

Trying to solve data missing issue from `open_mfdataset` with USDM.

### **`spi_quantiles_qa.ipynb`**

06.01.2022

In `analysis/usdm_spi_investigate_temporal.ipynb` found that D4 appears very overrepresented in the stacked timeseries for WA caf, so here I'm looking into what's up with that.

### **`test_drought_event_plot`**
7.14.2022

Following the work developed in `explore/drought_event_tracking.ipynb`, this notebook summarizes the methodology and provides space for analysis.

### **`usdm_spi_qa_sumzero_caf.ipynb`**

05.16.2022

Checking that the category area fractions sum to unity.

**Produces**: `ndrought_products/paired_USDM_SPI_caf.nc`

------------------------------------------

## query

Retrieving data from other sources such as THREDDS.

### **`collect_USDM_raster.ipynb`**

Date unknown.

Collect gridded USDM data, converting from geotifs to netcdfs. 

**Produces**: `usdm/Global/USDM_*.nc`, `usdm/Global/USDM_20000104_20220418.nc`

### **`connect_to_opendap.ipynb`**

11.18.2021, (updates 01.06.2022 and 03.02.2022)

Getting data from OPeNDAP on the THREDDS server at NKN. Note that some of the metrics may actually be produced from the bash script `/pool0/home/steinadi/data/drought/drought_impact/scripts/download_droughtvar_opendap.sh` instead.

**Produces**: `vegdri.nc`, `mcdi.nc`, `leri.nc`, `grace.nc`, `fdsi.nc`, `esi_4wk.nc`, `esi_12wk.nc`, `pdsi.nc`, `spi*.nc` 

### **`query_format_GRACE.ipynb`**
8.29.2022

Original data gathered from THREDDS was already categorized, so here I am getting the original matlab file from Katherine and getting it to work for me.

### **`query_huc_boundaries.ipynb`**

Date unknown.

Gather HUC boundaries.

**Produces**: `geometry/huc4.geojson`, `geometry/huc8.geojson`

### **`query_nps_boundaries.ipynb`**

01.18.2022

Gather national park service boundaries.

**Produces**: `geometry/pw_nps_bounds.geojson`, `geometry/pw_nps_select_geo.geojson`

### **`query_state_park_boundaries.ipynb`**

02.17.2022

Gather state park boundaries for WA and OR. (Correction: OR state parks was not completed)

**Produces**: `geometry/wa_state_park_bounds.geojson`.