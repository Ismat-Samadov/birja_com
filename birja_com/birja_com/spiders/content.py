import scrapy


class ContentSpider(scrapy.Spider):
    name = "content"
    allowed_domains = ["birja.com"]
    start_urls = ["https://birja.com/ads/az/476693/"]

    def parse(self, response):
        yield {
            'phone': response.xpath('//td[a[contains(@href, "tel:")]]/a/@href').get()
        }
