from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Template filter để lấy giá trị từ một từ điển (dictionary) theo khóa (key).
    Sử dụng trong template như sau: {{ my_dict|get_item:key_name }}
    """
    return dictionary.get(key, '') 