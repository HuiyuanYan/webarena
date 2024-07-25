import json
from typing import List
class Grounding:
    """
    Represents a visual grounding with attributes such as index, text, tag, position, and size.

    Attributes:
        index (int): The index of the visual grounding.
        text (str): The text associated with the visual grounding.
        tag (str): The tag of the visual grounding.
        left (float): The left position of the visual grounding.
        top (float): The top position of the visual grounding.
        width (float): The width of the visual grounding.
        height (float): The height of the visual grounding.
    """

    def __init__(self, backend_id: int, static_text: str, tag: str, union_bound:List[float]):
        """
        Initializes a new Grounding object.

        Args:
            index (int): The index of the visual grounding.
            text (str): The text associated with the visual grounding.
            tag (str): The tag of the visual grounding.
            union_bound(list): The position of the visual grounding:[left,top,width,height]
            left (float): The left position of the visual grounding.
            top (float): The top position of the visual grounding.
            width (float): The width of the visual grounding.
            height (float): The height of the visual grounding.
        """
        self.backend_id = backend_id
        self.static_text = static_text
        self.tag = tag
        self.union_bound = union_bound

    @classmethod
    def from_dict(cls, data: dict) -> 'Grounding':
        """
        Creates a Grounding object from a dictionary.

        Args:
            data (dict): A dictionary containing attributes of the visual grounding.

        Returns:
            Grounding: A new Grounding object initialized with data from the dictionary.
        """
        return cls(
            backend_id=data.get('backend_id', 0),
            static_text=data.get('static_text', ''),
            tag=data.get('tag', ''),
            union_bound = data.get('union_bound',[])
        )
    
    def to_json(self) -> dict:
        """
        Converts the Grounding object to a JSON-serializable dictionary.

        Returns:
            dict: A dictionary representation of the Grounding object.
        """
        return {
            'backend_id': self.backend_id,
            'static_text': self.static_text,
            'tag': self.tag,
            'union_bound':self.union_bound
        }
    
    def to_dict(self) -> dict:
        return self.to_json()
    
    @staticmethod
    def from_json(json_str: str):
        """
        Creates a Grounding object from a JSON string.

        Args:
            json_str (str): A JSON-formatted string representing the visual grounding.

        Returns:
            Grounding: A new Grounding object initialized with data from the JSON string.
        """
        data = json.loads(json_str)
        return Grounding.from_dict(data)

    def __str__(self):
        return f"[{self.backend_id}] {self.tag} '{self.static_text}'"
