# Socio-Economic Drought Impacts (SEDI)

Proposal Title: Improving Drought Indicators to Support Drought Impact Mitigation for Natural Resource Managment    
Funding oppertunity: CPO 2020, NOAA-OAR-CPO-2020-2006076, 280819, NIDIS-Coping with Drought    

Researchers: adi stein, Bart Nijssen, Katherine Hegewisch, John Abatzoglou    
README author: adi stein

Updated: 12.22.2021    

------------------------------------------

## Purpose & Persepctive

The purpose of this document is not to summarize the already written proposal but to describe the current state of the project and my, (adi, the person writing this), thought process and goals. Any concerns about the status of this project after checking if this document has been updated recently should be directed to my email: steinadi@uw.edu

A note on perspective: I am a graduated student at the UW studying Hydrology through Civil Engineering with influences from applied mathematics and gender studies. I am a white researcher belonging to a few non-majority communities and am working to document my though process throughout this project to hopefully increase the accessiblity of the project's technical details. Therefore, unless stated otherwise the opinions expressed in this repository are mine, influence by the world around me but not those submitted by group consensus. You can find out more about me by emailing me.

-------------------------------------------

## State of Project

Was examining national park data in comparison to FDSI. This resulted in no correlation nor shared characteristics among the distributions at Mt. Rainier National Park. Given the time period incorporated a range of values for FDSI, and none of the other national parks had values varying that drastically in shape from Mt. Rainier National Park, I am hesitant to spend more time looking for park shape files just to compare FDSI more. Aggregation metrics such as mean, median, max, and min where used to characterize both FDSI and the visitation data, yet neither yielded a notable pattern. It would be then more worthwhile to examine how the different drought indicators used (including those that are purely hydrologic or meteorologic but still used by stakeholders) differ before returning to look at a relationship with park visitation.

I am also curious about the timing of the various declarations and permits ... specifically if there is a drought threshold reached that commonly corresponds to a deceleration or permit being issued. I'd also be curious to see how the indicator classifying a drought and policy classifying a drought might differ to understand if there is a signal delay (more or less). I think the ban data and fire danger rating data could be useful for this given how several stakeholders mention they look at such data to determine business decisions.

To incorporate fish & wildlife, or water recreational activities in general, could look at SRI and attempt to abstract a lower threshold that would mean harm to the fish that might also impact fishing profits or make water recreational activities inoperable. Similar could be done perhaps for snowpack and snow recreation.

Lastly we need more socio-economic data beyond park visitations since I am thinking that the drought signal gets lost among all the other signals influencing park visitations. The report mentioned in goals uses QCEW ... so that would be worthwhile to pull from. Perhaps if there is some common insurance that folks are using when they can't run their business due to drought that would also be useful. 

I need to do more analysis, but am starting to have a sense of how this data could come together and form something interesting ... especially if we do a bit of brute force analysis involving throwing every indicator/metric used at the wall and seeing how they fair ...

## Goals
- compare drought indicators and proxy metrics to each other (mayhaps in the spirit of Hao et al. with a [multivariate approach](http://dx.doi.org/10.1016/j.advwatres.2013.03.009))
- examine drought onset timing and their relativity to drought declarations
- examine burn ban timing 
- incorporate statistics such as those expressed [here](https://www.drought.gov/documents/analysis-impact-drought-agriculture-local-economies-public-health-and-crime-across)

