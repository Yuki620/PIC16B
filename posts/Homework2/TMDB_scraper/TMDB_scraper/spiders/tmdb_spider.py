# to run 
# scrapy crawl tmdb_spider -o movies.csv -a subdir=671-harry-potter-and-the-philosopher-s-stone

import scrapy
from scrapy.selector import Selector


class TmdbSpider(scrapy.Spider):
    name = 'tmdb_spider'
    def __init__(self, subdir=None, *args, **kwargs):
        self.start_urls = [f"https://www.themoviedb.org/movie/{subdir}/"]
    
    def parse(self, response):
        """Navigates from a movie page to its 'Full Cast & Crew' page."""
        cast_page = response.url + '/cast'
        yield scrapy.Request(cast_page, callback=self.parse_full_credits)
    
    def parse_full_credits(self, response):
        """Yields requests for each actor's page from the 'Full Cast & Crew' page."""
        # extract the links for each actor
        actor_links = response.css('ol.people.credits:not(.crew) li div.info a::attr(href)').extract() 

        for link in actor_links:
            # use response.url join to get the absolute link!
            full_url = response.urljoin(link)
            #print(full_url)
            yield scrapy.Request(full_url, callback = self.parse_actor_page)
    
    def parse_actor_page(self, response):
        """Yields the actor's name and the names of movies or TV shows they've acted in."""
        # extract actor name
        actor_name = response.css('h2.title a::text').get().strip()
        
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
        
        



