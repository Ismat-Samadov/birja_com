import scrapy


class LinksSpider(scrapy.Spider):
    name = "links"
    allowed_domains = ["birja.com"]
    start_urls = ["https://birja.com/category/az/1/2/0/satilir/1"]

    def parse(self, response):
        # Extract the href attributes from anchor elements on the current page
        links = response.css('div.cs_card_title a::attr(href)').getall()

        for link in links:
            yield {
                'Link': link,
            }

        # Extract the next page link
        next_page = response.css('ul.pagination li.cs_active_pagination + li a::attr(href)').get()

        if next_page:
            yield response.follow(next_page, self.parse)
