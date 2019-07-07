from typing import List, Dict
import random


class Cuisine:
    def __init__(self, lines: List[str]):
        self.synonym = ''
        self.ingredient = ''
        self.recipe = ''
        self.attrib = ''
        name_str = lines[0]
        assert name_str[0] == '【' and name_str[-1] == '】'
        self.name = name_str[1:-1]
        for idx, line in enumerate(lines):
            if line == '[同義語]':
                self.synonym = lines[idx + 1]
            elif line == '[材料]':
                self.ingredient = lines[idx + 1]
            elif line == '[調理法]':
                self.recipe = lines[idx + 1]
            elif line == '[属性]':
                self.attrib = lines[idx + 1]

    def __str__(self):
        return self.name


class Food:
    def __init__(self, db_path: str):
        self.menu: Dict[str, Cuisine] = {}
        with open(db_path, encoding='utf-8', errors='replace') as f:
            lines = []
            for line in f:
                if line != '----------\n':
                    lines.append(line.strip())
                else:
                    cousine = Cuisine(lines)
                    self.menu[cousine.name] = cousine
                    lines = []

    def __getitem__(self, item: str):
        return self.menu[item]

    def __len__(self):
        return len(self.menu)


if __name__ == '__main__':
    food = Food('/Users/NobuhiroUeda/Data/food/foodkb.dictionary.txt')
    menues = list(food.menu.values())
    names = list(food.menu.keys())
    print(menues[random.randrange(387)])
    print(len(food))
