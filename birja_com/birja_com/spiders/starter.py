import scrapy


class StarterSpider(scrapy.Spider):
    name = "starter"
    allowed_domains = ["birja.com"]
    start_urls = ["https://birja.com/all_category/az"]

    def parse(self, response):
        for href in response.css('div.row a[href]::attr(href)').getall():
            yield {
                'href': f'birja.com{href}'
            }
