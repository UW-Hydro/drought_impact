# Socio-Economic Drought Impacts (SEDI)

Proposal Title: Improving Drought Indicators to Support Drought Impact Mitigation for Natural Resource Managment    
Funding oppertunity: CPO 2020, NOAA-OAR-CPO-2020-2006076, 280819, NIDIS-Coping with Drought    

Researchers: adi stein, Bart Nijssen, Katherine Hegewisch, John Abatzoglou    
README author: adi stein

Updated: 4.25.2022    

------------------------------------------

## Purpose & Persepctive

The purpose of this document is not to summarize the already written proposal but to describe the current state of the project and my, (adi, the person writing this), thought process and goals. Any concerns about the status of this project after checking if this document has been updated recently should be directed to my email: steinadi@uw.edu

A note on perspective: I am a graduated student at the UW studying Hydrology through Civil Engineering with influences from applied mathematics and gender studies. I am a white researcher belonging to a few non-majority communities and am working to document my though process throughout this project to hopefully increase the accessiblity of the project's technical details. Therefore, unless stated otherwise the opinions expressed in this repository are mine, influence by the world around me but not those submitted by group consensus. You can find out more about me by emailing me.

-------------------------------------------

## State of Project

4.25.2022

A little late of an update, but effectively this focus of the project is being dropped as a signal between national park data and drought was not clear nor really identifiable, so we moved on to the USDM evaluation focus, see the new README for more info.

3.2.2022

Recently I was delving into the Twitter API in attempts to use geo-tagged Tweet frequency as a proxy for visitation at our sites of interest, following a similar methodology to [Wilkins et al.](https://www.nature.com/articles/s41598-021-82145-z) with Flickr data. I have developed a workflow to gather data that has geo-spatial tags throughout the entire Twitter archive (since 2006) but there are unfortunately not many geo-tagged tweets that have the keywords I was using (national park names). The geo-tagging is important to be able to associate the tweet with actually being at the park (instead of just talking about the park or sharing a news story about it). As far as I can tell, (they do not have very descriptive documentation), other papers that I have seen do similar work are selecting social media data through their geo-tags around there are of interest (as I did). I might be able to do some analysis on how often each recreation site is mentioned throughout the year (similar to something like Google Trends), but it would not have a spatial dimension and I am not familiar with how to filter out noise for that. So I am going to move on and do analysis for the national and state parks with the data we already have since I was able to find geometries for them, then perhaps try social media data down the road if it becomes relevant again. This exploration still felt worthwhile given the various papers I looked at and now having more knowledge about the process. 

Regarding the IMPLAN data, I did end up checking it out and was unable to find data that would be specific enough to the water recreation sector.

Now I am taking a step back from drought metrics to find correlations with temperature as this has been promising in other works (including found in a quick plot by John). Once that is developed I can work on adding more information to try and connect visitations and licenses to drought signals. This will likely involve examining different time scales.

I am also going to be doing some data reorganizing, which make break some paths but they should be easily repairable if needed in the future.

Lastly, Katherine is working on compiling some fire burn data that we found through [WFAS](http://www.wfas.net/) ... which should provide us with a more promising historic record of fire danger rating. 

2.4.2022

This past Tuesday was the Coping With Drought PI meeting, which was helpful to reflect on a few different things for this project. It seemed that nearly every team was struggling with impacts. Some managed to identify hazards that could lead to impacts but impacts, being material consequences that people experience from drought, were slipping from people's hands for the most part. Analyzing hazards presents the opportunity to look at the root of impacts and attempt to forecast them, but without confirmation that that is folks experience outside the ivory tower, it's still not much better than where we started. I say that not to disqualify anyone's work as I am also in the same place of not having much stakeholder interaction, but to acknowledge the difference between impacts and hazards and be cautious of the conclusions we draw from them. Considering this puts into perspective the scientific drought indicators I've been looking at.

The CWD meeting also brought to light that it'll likely be next-to-impossible to get economic data disaggregated to the point of specific industries such as rafting. So I can do some more general analysis with the higher up patterns, but looking closer doesn't look likely ... which is sadly what I need to connect some dots. [IMPLAN](https://support.implan.com/hc/en-us/articles/115009674428-IMPLAN-Sectoring-NAICS-Correspondences) was suggested by Alvar from Public Policy Institute of California (PPIC), and John said he may know someone who uses it a lot ... so I'll look into it and see if I actually want to use it or not.

In conversation with Katherine about FDSI, it's impact on park visitation may be lagged or have some form of cumulative effect ... I should look at how FDSI visably shows up on trees or impacts canopy cover if I want to pursue that. If possible, we might be able to mine social media to look for trends in discussion about different parks/businesses to better understand what makes an experience favorable/unfavorable and therefore influence business in the future by a word-of-mouth model. 

Hunting and Fishing licenses from OR have also been gathered and organized now, I'm starting to look into what I can learn from them. Unfortunately they are only from 2016-2020 ... as with other data being rather receent too ... so I may need to consult with a statistician on how best to deal with short term data. Not to mention that COVID-19 has likely impacted data.

1.18.2022

I have managed to get more drought indicator data from OPeNDAP and well as become fairly familiar with how to manipulate it. Currently I am contemplating how to look at differences between the indicators ... but there is such a large spatial and temporal variation that makes aggregation tricky. For more scientific purposes I'll eventually need to evaluate how much data is lost in aggregation (think KL Divergence would be good for this) but for now I think I'll settle by reducing scale to just some national parks. Speaking of which, finally found some GIS data for the national parks, located [here](https://public-nps.opendata.arcgis.com/datasets/nps-boundary-1/explore?location=39.273308%2C-117.638430%2C5.00) ... so now I need to improve my querying ...

12.22.2021 Post

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

