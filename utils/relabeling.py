from collections import Counter

# Constants:
LOC_TAGS = ['DMN_COUNTRY', 'DMN_REGION', 'DMN_CITY', 'DMN_DISTRICT', 'DMN_STREET', 'DMN_HOUSE']
PER_TAGS = ['DMN_LAST_NAME', 'DMN_FIRST_NAME', 'DMN_MIDDLE_NAME']


# Utils:
def biolu2bio(tags: list) -> list:
    """ Relabel tags to BIO-format. Where:

    'B' - begging token,

    'I' - inner token.

    Which transformation need be apllied:

    'U' --> 'B'

    'L' --> 'I'

    :param tags: NER-tags for text tokens.
    :return: Relabeled NER-tokens.
    """
    return [f'B-{tag.split("-")[-1]}' if tag.split('-')[0] == 'U' else
            f'I-{tag.split("-")[-1]}' if tag.split('-')[0] == 'L' else
            tag for tag in tags]


def biolu2single_token(tags: list) -> list:
    """ Relabel tags to single token format. Here just needs to cut any token's prefixes.

        Example:

        ['B-DMN_COUNTRY', 'L-DMN_COUNTRY'] --> ['DMN_COUNTRY', 'DMN_COUNTRY']

        :param tags: NER-tags for text tokens.
        :return: Relabeled NER-tokens.
        """
    return [tag.split('-')[-1] for tag in tags]


def detailed2default(tags: list) -> list:
    """ Replace NER-tags to default: LOC and PER.

    Map of default tags:

    LOC: 'DMN_COUNTRY', 'DMN_REGION', 'DMN_CITY', 'DMN_DISTRICT', 'DMN_STREET', 'DMN_HOUSE'

    PER: 'DMN_LAST_NAME', 'DMN_FIRST_NAME', 'DMN_MIDDLE_NAME'

    :param tags: NER-tags for text tokens.
    :return: Relabeled NER-tokens.
    """
    return [f'{tag.split("-")[0]}-LOC' if tag.split('-')[-1] in LOC_TAGS else
            f'{tag.split("-")[0]}-PER' if tag.split('-')[-1] in PER_TAGS else
            tag for tag in tags]


def detailed2custom(tags: list, tag_map: dict) -> list:
    """ Replace NER-tags to custom defined by 'tag_map' argument. The main condition: replacing tags must not to
    intersect to each other or, in other words, lists (values) of 'tag_map' dict must not have intersections.

    Example:

    dict{
        'STREET': ['DMN_STREET', 'DMN_HOUSE'],
        'REGION': ['DMN_REGION', 'DMN_CITY', 'DMN_DISTRICT'],
        'LOC': ['DMN_LAST_NAME', 'DMN_FIRST_NAME', 'DMN_MIDDLE_NAME'],
    }

    :param tags: NER-tags for text tokens.
    :param tag_map: Dictionary of tag replacements what needed to be replaced.
    :return: Relabeled NER-tokens.
    """
    cnt = Counter()
    for k in tag_map:
        for tag in tag_map[k]:
            cnt.update({tag})
    intersected_tags = [k for k in cnt if cnt[k] > 1]
    if len(intersected_tags) > 0:
        print(f'Changes is not applied. You sent intersected map tags: {", ".join(intersected_tags)}')
        return tags
    new_tags = list()
    for tag in tags:
        new_tag = tag
        is_replaced = False
        for k in tag_map:
            if not is_replaced and tag.split('-')[-1] in tag_map[k]:
                new_tag = f'{tag.split("-")[0]}-{k}'
                is_replaced = True
        new_tags += [new_tag]
    return new_tags
