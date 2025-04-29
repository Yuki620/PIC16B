#!/usr/bin/env python
# coding: utf-8
---
title: "HW 2"
author: "Yuki Yu"
date: "2024-01-28"
categories: [Week 3, Homework]
---
# ## Web Scraping TMDB - "Wonka"

# In[45]:


import plotly.express as px
import pandas as pd
import numpy as np
import plotly.io as pio
pio.renderers.default='iframe'


# ### 1. Setting Up the Project

# ### 1.1 Pick a Movie 
# Pick your favorite movie, and locate its TMDB page by searching on https://www.themoviedb.org/. For example, I like the movie Wonka. Its TMDB page is at:
# 
#     https://www.themoviedb.org/movie/787699-wonka/
#     
# Save this URL for a moment.

# ### 1.2 Dry-Run Navigation
# Now, we’re just going to click through the navigation steps that our scraper will take.
# 
# First, click on the Full Cast & Crew link. This will take you to a page with URL of the form
# 
# \<original_url\>/cast
# 
# Next, scroll until you see the Cast section. Click on the portrait of one of the actors. This will take you to a page with a different-looking URL.
# 
# Finally, scroll down until you see the actor’s Acting section. Note the titles of a few movies and TV shows in this section.
# 
# Our scraper is going to replicate this process. Starting with your favorite movie, it’s going to look at all the actors in that movie, and then log all the other movies or TV shows that they worked on.
# 
# At this point, it would be a good idea for you to use the Developer Tools on your browser to inspect individual HTML elements and look for patterns among the names you are looking for.

# ### 1.3. Initialize Your Project
# Open a terminal and type:
# 
#     conda activate PIC16B
#     scrapy startproject TMDB_scraper
#     cd TMDB_scraper

# ### 1.4 Tweak Settings
# For now, add the following line to the file settings.py:

# In[27]:


CLOSESPIDER_PAGECOUNT = 20


# This line just prevents your scraper from downloading too much data while you’re still testing things out. You’ll remove this line later.

# ### Troubleshooting
# If you run into `403 Forbidden` errors from the website detecting that you're a bot, follow the following steps: 
# <br>
# 
# **Installed `scrapy_fake_useragent`** <br>
# Make sure that it is installed in the correct environment and location. <br>
# 
# **Add the following lines in `settings.py`**
# 

# This setting is used to specify the amount of time (in seconds) that the scraper should wait before downloading consecutive pages from the same website. A DOWNLOAD_DELAY helps in mimicking human browsing behavior more closely and reduces the risk of getting banned or blocked by the website's server for sending too many requests too quickly.

# In[28]:


DOWNLOAD_DELAY = 3


# Some websites use cookies to detect and block scrapers. If the website's functionality you are scraping does not require cookies, disabling them can simplify your scraping process. Setting COOKIES_ENABLED to False turns off cookie handling, meaning your scraper won't send or receive any cookies with the requests.

# In[ ]:


COOKIES_ENABLED = False


# The goal of these settings is to make the scraper mimic a real user's browsing behavior more closely and to improve its ability to access web pages by avoiding detection based on User-Agent patterns or being blocked due to repeated requests from the same User-Agent.

# In[29]:


DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
    'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 400,
    'scrapy_fake_useragent.middleware.RetryUserAgentMiddleware': 401,
}

FAKEUSERAGENT_PROVIDERS = [
    'scrapy_fake_useragent.providers.FakeUserAgentProvider',  # This is the first provider we'll try
    'scrapy_fake_useragent.providers.FakerProvider',  # If FakeUserAgentProvider fails, we'll use faker to generate a user-agent string for us
    'scrapy_fake_useragent.providers.FixedUserAgentProvider',  # Fall back to USER_AGENT value
]


# ### 2. Write Your Scraper

# Create a file inside the `spiders` directory called `tmdb_spider.py`. Add the following lines to the file:

# In[ ]:


# to run 
# scrapy crawl tmdb_spider -O results.csv -a subdir=787699-wonka

import scrapy

class TmdbSpider(scrapy.Spider):
    name = 'tmdb_spider'
    def __init__(self, subdir=None, *args, **kwargs):
        self.start_urls = [f"https://www.themoviedb.org/movie/{subdir}/"]


# Then, you will be able to run your completed spider for a movie of your choice by giving its subdirectory on TMDB website as an extra command-line argument.

# Now implement the following 3 parsing methods in the `TmdbSpider` class as well:

# `parse(self, response)` should assume that you start on a movie page, and then navigate to the Full Cast & Crew page. Remember that this page has url <movie_url>cast. (You are allowed to hardcode that part.) Once there, the parse_full_credits(self,response) should be called, by specifying this method in the callback argument to a yielded scrapy.Request. The parse() method does not return any data. This method should be no more than 5 lines of code, excluding comments and docstrings.

# In[ ]:


def parse(self, response):
    """Navigates from a movie page to its 'Full Cast & Crew' page."""
    cast_page = response.url + '/cast'
    yield scrapy.Request(cast_page, callback=self.parse_full_credits)


# `parse_full_credits(self, response)` should assume that you start on the Full Cast & Crew page. Its purpose is to yield a scrapy.Request for the page of each actor listed on the page. Crew members are not included. The yielded request should specify the method parse_actor_page(self, response) should be called when the actor’s page is reached. The parse_full_credits() method does not return any data. This method should be no more than 5 lines of code, excluding comments and docstrings.

# In[ ]:


def parse_full_credits(self, response):
    """Yields requests for each actor's page from the 'Full Cast & Crew' page."""
    # extract the links for each actor
    actor_links = response.css('ol.people.credits:not(.crew) li a::attr(href)').extract() 

    for link in actor_links:
        # use response.urljoin to get the absolute link!
        full_url = response.urljoin(link)
        yield scrapy.Request(full_url, callback = self.parse_actor_page)


# `parse_actor_page(self, response)` should assume that you start on the page of an actor. It should yield a dictionary with two key-value pairs, of the form `{"actor" : actor_name, "movie_or_TV_name" : movie_or_TV_name}`. The method should yield one such dictionary for each of the movies or TV shows on which that actor has worked in an acting role. Note that you will need to determine both the name of the actor and the name of each movie or TV show. This method should be no more than 15 lines of code, excluding comments and docstrings.

# In[ ]:


def parse_actor_page(self, response):
    """Yields the actor's name and the names of movies or TV shows they've acted in."""
    # extract actor name
    actor_name = response.css('h2.title a::text').get().strip()
        
    # Make sure we only select the 'Acting'
    h3_elements = response.css('div.credits_list h3')
    for h3 in h3_elements:
        if 'Acting' in h3.xpath('./text()').get():
            acting_table = h3.xpath('following-sibling::table[1]').get()
            break
    table_selector = Selector(text=acting_table)

    for credit in table_selector.css('table.credit_group tr'):
        # extract movie or tv show name
        movie_or_TV_name = credit.css('td.role a.tooltip bdi::text').get().strip()
        yield {
            'actor': actor_name,
            'movie_or_TV_name': movie_or_TV_name
            }


# Provided that these methods are correctly implemented, you can run the command

# In[ ]:


scrapy crawl tmdb_spider -o results.csv -a subdir=787699-wonka


# to create a `.csv` file with a column for actors and a column for movies or TV shows for "Wonka" (-o to append, and -O to overwrite file).

# ### 3. Make Your Recommendations
# Once your spider is fully written, comment out the line

# In[33]:


CLOSESPIDER_PAGECOUNT = 20


# in the `settings.py` file. Then, the command

# In[ ]:


scrapy crawl tmdb_spider -O results.csv -a subdir=787699-wonka


# will run your spider and save a CSV file called `results.csv`, with columns for actor names and the movies and TV shows on which they featured in.
# 
# Once you’re happy with the operation of your spider, compute a sorted list with the top movies and TV shows that share actors with your favorite movie or TV show.
# 
# **Prepare the Table**

# In[41]:


df = pd.read_csv('results.csv')
df = df.groupby('movie_or_TV_name').size().reset_index(name='number of shared actors')
df.head()


# **Sort the Table** <br>
# Since "Wonka" would obviously have the highest amount of shared actors, we will exclude it from our recommendation table.

# In[44]:


df = df.sort_values(by='number of shared actors', ascending=False)
df.index = range(0, len(df))
df = df.iloc[1:11,]
df


# **Make the Bar Chart with Plotly**

# In[35]:


fig = px.bar(df, x='movie_or_TV_name', y='number of shared actors' 
            ,title="Recommendations after \"Wonka\""
            ,labels={
                "movie_or_TV_name": "Movie or TV Name",
                "number of shared actors": "Number of Shared Actors"
            })
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
fig.show()

