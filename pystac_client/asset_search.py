import json
from copy import deepcopy
from datetime import timezone, datetime as datetime_
from typing import Union, Optional, Iterable, List, Tuple, Iterator, Dict, Any

from functools import lru_cache
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
from pystac import Item, Link, Collection
from pystac import Asset as PystacAsset

from pystac_client.item_search import (
    DATETIME_REGEX,
    DatetimeOrTimestamp,
    Datetime,
    DatetimeLike,
    BBox,
    BBoxLike,
    Collections,
    CollectionsLike,
    IDs,
    IDsLike,
    Intersects,
    IntersectsLike,
    QueryLike,
    FilterLangLike,
    FilterLike,
    Fields,
    FieldsLike,
    Sortby,
    SortbyLike,
    StacApiIO,
    dict_merge,
    FreeTextLike
)
from pystac_client.conformance import ConformanceClasses
from pystac.stac_io import StacIO
from pystac_client.stac_api_io import StacApiIO


Items = Tuple[str, ...]
ItemsLike = Union[List[str], Iterator[str], str]


class Asset(PystacAsset):

    type: str
    properties: Optional[Union[str, Dict]]
    item: str
    stac_version: str
    stac_extension: Optional[List]
    asset_id: str
    size: Optional[int]
    links: List[Link]

    def __init__(
            self,
            href: str,
            item: str,
            type: str,
            asset_id: str,
            properties: Optional[Union[str, Dict]] = None,
            stac_version: str = None,
            stac_extension: Optional[List] = None,
            size: Optional[int] = None,
            links: List[Link] = None,
            title: Optional[str] = None,
            description: Optional[str] = None,
            media_type: Optional[str] = None,
            roles: Optional[List[str]] = None,
            extra_fields: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super().__init__(href, title, description, media_type, roles, extra_fields)
        self.type = type
        self.properties = properties
        self.item = item
        self.stac_version = stac_version
        self.stac_extension = stac_extension
        self.asset_id = asset_id
        self.size = size
        self.links = links


class AssetSearch:

    def __init__(self,
                 url: str,
                 *,
                 limit: Optional[int] = 100,
                 bbox: Optional[BBoxLike] = None,
                 datetime: Optional[DatetimeLike] = None,
                 intersects: Optional[IntersectsLike] = None,
                 ids: Optional[IDsLike] = None,
                 query: Optional[QueryLike] = None,
                 filter: Optional[FilterLike] = None,
                 filter_lang: Optional[FilterLangLike] = None,
                 sortby: Optional[SortbyLike] = None,
                 fields: Optional[FieldsLike] = None,
                 max_assets: Optional[int] = None,
                 method: Optional[str] = 'POST',
                 stac_io: Optional[StacIO] = None,
                 client: Optional["Client"] = None,
                 q: Optional[FreeTextLike] = None,):
        self.url = url
        self.client = client

        if stac_io:
            self._stac_io = stac_io
        else:
            self._stac_io = StacApiIO()
        self._stac_io.assert_conforms_to(ConformanceClasses.ASSET_SEARCH)

        self._max_assets = max_assets
        if self._max_assets is not None and limit is not None:
            limit = min(limit, self._max_assets)

        if limit is not None and (limit < 1 or limit > 10000):
            raise Exception(f"Invalid limit of {limit}, mist be between 1 and 10,000")

        self.method = method

        params = {
            'limit': limit,
            'bbox': self._format_bbox(bbox),
            'datetime': self._format_datetime(datetime),
            'ids': self._format_ids(ids),
            'intersects': self._format_intersects(intersects),
            'query': self._format_query(query),
            'filter': self._format_filter(filter),
            'filter-lang': self._format_filter_lang(filter, filter_lang),
            'sortby': self._format_sortby(sortby),
            'fields': self._format_fields(fields),
            "q": self._format_freetext(q),
        }

        self._parameters = {k: v for k, v in params.items() if v is not None}

    def get_parameters(self):
        if self.method == 'POST':
            return self._parameters
        elif self.method == 'GET':
            params = deepcopy(self._parameters)
            if 'bbox' in params:
                params['bbox'] = ','.join(map(str, params['bbox']))
            if 'ids' in params:
                params['ids'] = ','.join(params['ids'])
            if 'collections' in params:
                params['collections'] = ','.join(params['collections'])
            if 'intersects' in params:
                params['intersects'] = json.dumps(params['intersects'])
            return params
        else:
            raise Exception(f"Unsupported method {self.method}")

    @staticmethod
    def _format_query(value: List[QueryLike]) -> Optional[dict]:
        if value is None:
            return None

        OP_MAP = {'>=': 'gte', '<=': 'lte', '=': 'eq', '>': 'gt', '<': 'lt'}

        if isinstance(value, list):
            query = {}
            for q in value:
                for op in ['>=', '<=', '=', '>', '<']:
                    parts = q.split(op)
                    if len(parts) == 2:
                        param = parts[0]
                        val = parts[1]
                        if param == "gsd":
                            val = float(val)
                        query = dict_merge(query, {parts[0]: {OP_MAP[op]: val}})
                        break
        else:
            query = value

        return query

    def _format_filter_lang(self, filter: FilterLike, value: FilterLangLike) -> Optional[str]:
        if filter is None:
            return None

        if value is None:
            return 'cql-json'

        return value

    def _format_filter(self, value: FilterLike) -> Optional[dict]:
        if value is None:
            return None

        self._stac_io.assert_conforms_to(ConformanceClasses.FILTER)
        return value

    @staticmethod
    def _format_bbox(value: Optional[BBoxLike]) -> Optional[BBox]:
        if value is None:
            return None

        if isinstance(value, str):
            bbox = tuple(map(float, value.split(',')))
        else:
            bbox = tuple(map(float, value))

        return bbox

    @staticmethod
    def _format_datetime(value: Optional[DatetimeLike]) -> Optional[Datetime]:
        def _to_utc_isoformat(dt):
            dt = dt.astimezone(timezone.utc)
            dt = dt.replace(tzinfo=None)
            return dt.isoformat("T") + "Z"

        def _to_isoformat_range(component: DatetimeOrTimestamp):
            """Converts a single DatetimeOrTimestamp into one or two Datetimes.

            This is required to expand a single value like "2017" out to the whole year. This function returns two values.
            The first value is always a valid Datetime. The second value can be None or a Datetime. If it is None, this
            means that the first value was an exactly specified value (e.g. a `datetime.datetime`). If the second value is
            a Datetime, then it will be the end of the range at the resolution of the component, e.g. if the component
            were "2017" the second value would be the last second of the last day of 2017.
            """
            if component is None:
                return "..", None
            elif isinstance(component, str):
                if component == "..":
                    return component, None

                match = DATETIME_REGEX.match(component)
                if not match:
                    raise Exception(f"invalid datetime component: {component}")
                elif match.group("remainder"):
                    if match.group("tz_info"):
                        return component, None
                    else:
                        return f"{component}Z", None
                else:
                    year = int(match.group("year"))
                    optional_month = match.group("month")
                    optional_day = match.group("day")

                if optional_day is not None:
                    start = datetime_(year,
                                      int(optional_month),
                                      int(optional_day),
                                      0,
                                      0,
                                      0,
                                      tzinfo=tzutc())
                    end = start + relativedelta(days=1, seconds=-1)
                elif optional_month is not None:
                    start = datetime_(year, int(optional_month), 1, 0, 0, 0, tzinfo=tzutc())
                    end = start + relativedelta(months=1, seconds=-1)
                else:
                    start = datetime_(year, 1, 1, 0, 0, 0, tzinfo=tzutc())
                    end = start + relativedelta(years=1, seconds=-1)
                return _to_utc_isoformat(start), _to_utc_isoformat(end)
            else:
                return _to_utc_isoformat(component), None

        if value is None:
            return None
        elif isinstance(value, datetime_):
            return _to_utc_isoformat(value)
        elif isinstance(value, str):
            components = value.split("/")
        else:
            components = list(value)

        if not components:
            return None
        elif len(components) == 1:
            start, end = _to_isoformat_range(components[0])
            if end is not None:
                return f"{start}/{end}"
            else:
                return start
        elif len(components) == 2:
            start, _ = _to_isoformat_range(components[0])
            backup_end, end = _to_isoformat_range(components[1])
            return f"{start}/{end or backup_end}"
        else:
            raise Exception(
                f"too many datetime components (max=2, actual={len(components)}): {value}")

    @staticmethod
    def _format_items(value: Optional[ItemsLike]) -> Optional[Items]:
        def _format(i):
            if isinstance(i, str):
                return i
            if isinstance(i, Iterable):
                return tuple(map(_format, i))

            return i.id

        if value is None:
            return None
        if isinstance(value, str):
            return tuple(map(_format, value.split(',')))
        if isinstance(value, Collection):
            return _format(value),

        return _format(value)

    @staticmethod
    def _format_ids(value: Optional[IDsLike]) -> Optional[IDs]:
        if value is None:
            return None

        if isinstance(value, str):
            return tuple(value.split(','))

        return tuple(value)

    def _format_sortby(self, value: Optional[SortbyLike]) -> Optional[Sortby]:
        if value is None:
            return None

        self._stac_io.assert_conforms_to(ConformanceClasses.SORT)

        if isinstance(value, str):
            return tuple(value.split(','))

        return tuple(value)

    def _format_fields(self, value: Optional[FieldsLike]) -> Optional[Fields]:
        if value is None:
            return None

        self._stac_io.assert_conforms_to(ConformanceClasses.FIELDS)

        if isinstance(value, str):
            return tuple(value.split(','))

        return tuple(value)

    @staticmethod
    def _format_intersects(value: Optional[IntersectsLike]) -> Optional[Intersects]:
        if value is None:
            return None
        if isinstance(value, str):
            return json.loads(value)
        return deepcopy(getattr(value, '__geo_interface__', value))

    def _format_freetext(self, q: Optional[FreeTextLike]) -> Optional[str]:
        if q is not None:
            self._stac_io.assert_conforms_to(ConformanceClasses.FREETEXT)
            return q
        else:
            return None

    def get_assets(self) -> Iterator[Dict]:
        for page in self._stac_io.get_pages(self.url, self.method, self.get_parameters()):
            for asset in page['features']:
                yield Asset(**asset)

    @lru_cache(1)
    def matched(self) -> Optional[int]:
        """Return number matched for search

        Returns the value from the `numberMatched` or `context.matched` field.
        Not all APIs will support counts in which case a warning will be issued

        Returns:
            int: Total count of matched items. If counts are not supported `None`
            is returned.
        """
        params = {**self.get_parameters(), "limit": 1}
        resp = self._stac_io.read_json(self.url, method=self.method, parameters=params)
        found = None
        if "context" in resp:
            found = resp["context"]["matched"]
        elif "numberMatched" in resp:
            found = resp["numberMatched"]
        if found is None:
            warnings.warn("numberMatched or context.matched not in response")
        return found

