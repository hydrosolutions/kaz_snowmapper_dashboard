TRANSLATIONS = {
    'en': {
        'title': 'Tajikistan Snowmapper Dashboard',
        'variable_select': 'Select Variable',
        'date_select': 'Select Date',
        'language': 'Language',
        'map_type': 'Map Type',
        'opacity': 'Layer Opacity',
        'snow_height': 'Snow Height',
        'hs_short': 'HS',
        'swe': 'Snow Water Equivalent',
        'swe_short': 'SWE',
        'runoff': 'Snow melt',
        'rof_short': 'SM',
        'change_24h': '24h Change',
        'change_48h': '48h Change',
        'change_72h': '72h Change',
        'loading': 'Loading data...',
        'no_data': 'No data available',
        'unit_mm': 'mm',
        'unit_m3s': 'm³/s'
    },
    'ru': {
        'title': 'Панель мониторинга снега Таджикистана',
        'variable_select': 'Select Variable ru',
        'date_select': 'Select Date ru',
        'language': 'Language ru',
        'map_type': 'Map Type ru',
        'opacity': 'Layer Opacity ru',
        'snow_height': 'Snow Height ru',
        'hs_short': 'HS ru',
        'swe': 'Snow Water Equivalent ru',
        'swe_short': 'SWE ru',
        'runoff': 'Snow melt ru',
        'rof_short': 'SM ru',
        'change_24h': '24h Change ru',
        'change_48h': '48h Change ru',
        'change_72h': '72h Change ru',
        'loading': 'Loading data... ru',
        'no_data': 'No data available ru',
        'unit_mm': 'mm ru',
        'unit_m3s': 'm³/s ru'
    },
    'tj': {
        'title': 'Лавҳаи назорати барф Тоҷикистон',
        'variable_select': 'Select Variable tj',
        'date_select': 'Select Date tj',
        'language': 'Language tj',
        'map_type': 'Map Type tj',
        'opacity': 'Layer Opacity tj',
        'snow_height': 'Snow Height tj',
        'hs_short': 'HS tj',
        'swe': 'Snow Water Equivalent tj',
        'swe_short': 'SWE tj',
        'runoff': 'Snow melt tj',
        'rof_short': 'SM tj',
        'change_24h': '24h Change tj',
        'change_48h': '48h Change tj',
        'change_72h': '72h Change tj',
        'loading': 'Loading data... tj',
        'no_data': 'No data available tj',
        'unit_mm': 'mm tj',
        'unit_m3s': 'm³/s tj'
    }
}

def get_text(key: str, language: str) -> str:
    """Get translated text for given key and language."""
    return TRANSLATIONS.get(language, TRANSLATIONS['en']).get(key, key)