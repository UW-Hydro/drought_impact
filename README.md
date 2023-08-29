# nDrought: Analysis of USDM capturing other forms of drought

Proposal Title: Improving Drought Indicators to Support Drought Impact Mitigation for Natural Resource Managment    
Funding oppertunity: CPO 2020, NOAA-OAR-CPO-2020-2006076, 280819, NIDIS-Coping with Drought    

Researchers: adi stein, Bart Nijssen, Katherine Hegewisch, John Abatzoglou    
README author: adi stein

Updated: 8.22.2023    

------------------------------------------

## Purpose & Persepctive

The purpose of this document is not to summarize the already written proposal but to describe the current state of the project and my, (adi, the person writing this), thought process and goals. Any concerns about the status of this project after checking if this document has been updated recently should be directed to my email: steinadi@uw.edu

A note on perspective: I am a graduate student at the UW studying Hydrology through Civil Engineering with influences from applied mathematics and gender studies. I am a white researcher belonging to a few non-majority communities and am working to document my though process throughout this project to hopefully increase the accessibility of the project's technical details. Therefore, unless stated otherwise the opinions expressed in this repository are mine, influence by the world around me but not those submitted by group consensus. You can find out more about me by emailing me.

-------------------------------------------

## Abstract - v.1

Drought definitions and drought metrics come in many different forms and it is not always clear which definition or metric is most useful for describing or forecasting drought impacts. Drought metrics range from those which only incorporate precipitation anomalies to more comprehensive measures such as the United States Drought Monitor (USDM), which incorporates on-the-ground observations of drought impacts to arrive at a consensus-based drought status. Here, we use a range of drought metrics, as reported in the Climate Toolbox (https://climatetoolbox.org/), to investigate their level of agreement in assessing drought conditions on the Columbia River Basin and evaluate what drought impacts they capture. 

We compare six drought metrics and evaluate how they spatially evolve over the two decade period from 2000 to 2021. Our drought metrics are the USDM, Standardized Precipitation Index, Standardized Precipitation Evaporation Index, Evaporative Demand Drought Index, Palmer Drought Severity Index, and soil moisture percentiles. By evaluating agree/disagreement of onset, duration, intensity, and spatial tracks at various time windows, we can develop generalized patterns between the metrics. 

Drought metrics will then be related to drought impacts determined from literature, news stories, and drought declarations. We evaluate how well each drought metric aligns with each impact, including checking for delayed connections between event and impact. This shifts focus from classifying drought with a singular metric to a nexus of metrics, giving some weight over others depending on the impact of concern. Documenting generalized patterns can make using this nexus more accessible in policy and aid decisions. In doing so, we characterize what types and manifestations of drought are represented in each metric and where decision making could be improved to better address drought impacts in the Columbia River Basin.

-------------------------------------------

## State of Project

8.22.2023

I haven't been as regular about these as I initially hoped ... lessons for the future. The project is now wrapping up and coming to a close. I've performed parameter testing that is to be published in a paper for Journal of Hydrometeorology. 

4.25.2022

Finally getting around to updating these README's and changing from SEDI to nDrought (name pending ... it's the best thing I could find for now). The experiment outline and question above (which yes, I realize does not have a question mark) captures what this is about. So I'm going to skip that explanation.

Currently I'm focusing on comparing SPI to USDM, developing the workflows to do so, (including clipping an unruly amount of gridded USDM data), and so forth.

1.3.2023

It has been a rather busy time since this last larger update. A wide variety of things have been developed, ultimately focusing on evolutionary comparison between metrics in a nexus instead of holding any singular metric as truth. This has led to a graph network sort of approach with the Drought Network. Work on the project was presented at the AGU Fall Meeting 2022.