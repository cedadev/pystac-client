# flake8: noqa

from pystac_client.version import __version__
from pystac_client.item_search import ItemSearch
from pystac_client.asset_search import AssetSearch
from pystac_client.client import Client
from pystac_client.collection_client import CollectionClient
from pystac_client.conformance import ConformanceClasses

# from typing import Iterator
#
#
# def get_assets(self) -> Iterator['Asset']:
#     asset_link = self.get_single_link('assets')
#     if asset_link is not None:
#         search = AssetSearch(asset_link.href, method='GET', stac_io=self.get_root()._stac_io)
#         yield from search.get_assets()