"""
Generate synthetic noisy STT transcripts with PII entities.
Outputs in exact format: {"id": "utt_XXX", "text": "...", "entities": [...]}
"""
import json
import random
from typing import List, Dict
from datetime import datetime, timedelta


class NoisySTTDataGenerator:
    """Generate realistic noisy STT transcripts with PII entities."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._init_entity_pools()
        self._init_templates()
        self._init_noise_patterns()
    
    def _init_entity_pools(self):
        """Initialize entity value pools."""
        self.first_names = [
            "john", "mary", "james", "jennifer", "michael", "linda", "robert", "patricia",
            "david", "barbara", "william", "elizabeth", "richard", "susan", "joseph",
            "jessica", "thomas", "sarah", "charles", "karen", "daniel", "nancy"
        ]
        
        self.last_names = [
            "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis",
            "rodriguez", "martinez", "hernandez", "lopez", "wilson", "anderson", "thomas",
            "taylor", "moore", "jackson", "martin", "lee", "perez", "thompson", "white"
        ]
        
        self.cities = [
            "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
            "san antonio", "san diego", "dallas", "san jose", "austin", "seattle",
            "denver", "boston", "atlanta", "miami", "portland", "detroit", "minneapolis"
        ]
        
        self.locations = [
            "central park", "times square", "main street", "first avenue", "oak boulevard",
            "pine street", "maple drive", "city hall", "public library", "community center",
            "shopping mall", "train station", "airport", "coffee shop", "restaurant"
        ]
        
        self.filler_words = [
            "um", "uh", "like", "you know", "i mean", "basically", "actually", "so", "well"
        ]
        
        self.email_domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.com"
        ]
    
    def _init_templates(self):
        """Initialize utterance templates."""
        self.templates = [
            # Credit card
            "my credit card number is {CREDIT_CARD}",
            "the card is {CREDIT_CARD}",
            "please charge {CREDIT_CARD}",
            "i want to use card {CREDIT_CARD}",
            
            # Phone
            "you can call me at {PHONE}",
            "my phone number is {PHONE}",
            "reach me at {PHONE}",
            "call {PHONE} for more info",
            
            # Email
            "send it to {EMAIL}",
            "my email is {EMAIL}",
            "contact me at {EMAIL}",
            "email me at {EMAIL}",
            
            # Person name
            "my name is {PERSON_NAME}",
            "this is {PERSON_NAME} calling",
            "im {PERSON_NAME}",
            "{PERSON_NAME} from sales",
            "{PERSON_NAME} will handle it",
            
            # Date
            "on {DATE}",
            "schedule for {DATE}",
            "meeting is {DATE}",
            "due {DATE}",
            
            # City
            "im in {CITY}",
            "traveling to {CITY}",
            "located in {CITY}",
            "from {CITY}",
            
            # Location
            "meet at {LOCATION}",
            "near {LOCATION}",
            "at {LOCATION}",
            
            # Multi-entity
            "hi im {PERSON_NAME} my email is {EMAIL}",
            "{PERSON_NAME} from {CITY} number is {PHONE}",
            "contact {PERSON_NAME} at {EMAIL} or {PHONE}",
            "card {CREDIT_CARD} expires {DATE}",
            "im {PERSON_NAME} call me at {PHONE} in {CITY}",
            "{PERSON_NAME} meet you at {LOCATION} on {DATE}",
        ]
    
    def _init_noise_patterns(self):
        """Initialize STT noise patterns."""
        self.digit_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
    
    def _generate_credit_card(self) -> str:
        """Generate credit card in noisy STT format."""
        # Generate 16-digit number
        digits = ''.join([str(random.randint(0, 9)) for _ in range(16)])
        
        # STT often outputs as groups or spelled out
        formats = [
            # Grouped format
            f"{digits[0:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}",
            # Some digits spelled out
            f"{digits[0:4]} {self._spell_some_digits(digits[4:8])} {digits[8:12]} {digits[12:16]}",
            # All spelled out (rare)
            ' '.join([self.digit_words[d] for d in digits[0:8]]) + f" {digits[8:16]}",
        ]
        return random.choice(formats)
    
    def _spell_some_digits(self, digits: str) -> str:
        """Spell out some digits."""
        if random.random() < 0.3:
            return ' '.join([self.digit_words[d] for d in digits])
        return digits
    
    def _generate_phone(self) -> str:
        """Generate phone number in noisy STT format."""
        area = str(random.randint(200, 999))
        prefix = str(random.randint(200, 999))
        line = str(random.randint(1000, 9999))
        
        # STT formats
        formats = [
            # Numbers with spaces
            f"{area} {prefix} {line}",
            # Area spelled, rest digits
            f"{self._spell_some_digits(area)} {prefix} {line}",
            # All digits no spaces
            f"{area}{prefix}{line}",
            # With "dash" or "hyphen" as words
            f"{area} dash {prefix} dash {line}",
        ]
        return random.choice(formats)
    
    def _generate_email(self) -> str:
        """Generate email in noisy STT format."""
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        domain = random.choice(self.email_domains)
        
        # STT email formats
        name_part = random.choice([
            f"{first} dot {last}",
            f"{first}{last}",
            f"{first} underscore {last}",
            f"{first} {random.randint(1, 99)}",
        ])
        
        # Domain part
        domain_part = domain.replace('.', ' dot ')
        
        return f"{name_part} at {domain_part}"
    
    def _generate_person_name(self) -> str:
        """Generate person name."""
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        
        if random.random() < 0.7:
            return f"{first} {last}"
        else:
            return first
    
    def _generate_date(self) -> str:
        """Generate date in noisy STT format."""
        base = datetime.now()
        days_offset = random.randint(-180, 180)
        date = base + timedelta(days=days_offset)
        
        formats = [
            # Spelled out month
            f"{date.strftime('%B')} {date.day}",
            f"{date.strftime('%B')} {date.day} {date.year}",
            # Numeric with "slash" as word
            f"{date.month} slash {date.day} slash {date.year}",
            # Day of week
            date.strftime('%A'),
            f"next {date.strftime('%A')}",
            # Relative
            "tomorrow",
            "today",
            "next week",
            "next month",
        ]
        return random.choice(formats)
    
    def _generate_city(self) -> str:
        """Generate city name."""
        return random.choice(self.cities)
    
    def _generate_location(self) -> str:
        """Generate location name."""
        return random.choice(self.locations)
    
    def _apply_stt_noise(self, text: str, entities: List[Dict]) -> tuple:
        """
        Apply realistic STT noise to text.
        Returns (noisy_text, adjusted_entities).
        """
        # Always lowercase (STT doesn't preserve case)
        text = text.lower()
        
        # Add filler words (randomly)
        if random.random() < 0.4:
            words = text.split()
            if len(words) > 2:
                pos = random.randint(1, len(words) - 1)
                filler = random.choice(self.filler_words)
                words.insert(pos, filler)
                text = ' '.join(words)
                
                # Adjust entity positions
                filler_len = len(filler) + 1  # +1 for space
                for ent in entities:
                    # Calculate position of inserted filler
                    words_before_insert = ' '.join(text.split()[:pos])
                    insert_pos = len(words_before_insert)
                    if insert_pos > 0:
                        insert_pos += 1  # account for space
                    
                    # Adjust entities that come after insertion
                    if ent['start'] >= insert_pos:
                        ent['start'] += filler_len
                        ent['end'] += filler_len
        
        # Remove some punctuation
        text = text.replace(',', '').replace('.', '')
        
        # Occasional word repetition
        if random.random() < 0.15:
            words = text.split()
            if len(words) > 2:
                pos = random.randint(0, len(words) - 1)
                words.insert(pos, words[pos])
                text = ' '.join(words)
                
                # Adjust entities (simplified - just shift everything after)
                repeat_pos = len(' '.join(words[:pos]))
                if repeat_pos > 0:
                    repeat_pos += 1
                repeat_len = len(words[pos]) + 1
                
                for ent in entities:
                    if ent['start'] >= repeat_pos:
                        ent['start'] += repeat_len
                        ent['end'] += repeat_len
        
        return text, entities
    
    def generate_example(self, example_id: int) -> Dict:
        """Generate a single training example."""
        template = random.choice(self.templates)
        
        # Track entities
        entities = []
        text = template
        
        # Entity generators
        generators = {
            'CREDIT_CARD': self._generate_credit_card,
            'PHONE': self._generate_phone,
            'EMAIL': self._generate_email,
            'PERSON_NAME': self._generate_person_name,
            'DATE': self._generate_date,
            'CITY': self._generate_city,
            'LOCATION': self._generate_location,
        }
        
        # Replace placeholders and track positions
        for entity_type, generator in generators.items():
            placeholder = f"{{{entity_type}}}"
            if placeholder in text:
                value = generator()
                start = text.find(placeholder)
                text = text.replace(placeholder, value, 1)
                end = start + len(value)
                
                entities.append({
                    'start': start,
                    'end': end,
                    'label': entity_type,
                })
        
        # Apply STT noise
        text, entities = self._apply_stt_noise(text, entities)
        
        # Sort entities by position
        entities.sort(key=lambda e: e['start'])
        
        return {
            'id': f"utt_{example_id:04d}",
            'text': text,
            'entities': entities,
        }
    
    def generate_dataset(self, num_examples: int, start_id: int = 0) -> List[Dict]:
        """Generate multiple examples."""
        return [
            self.generate_example(start_id + i)
            for i in range(num_examples)
        ]


def save_jsonl(data: List[Dict], filepath: str):
    """Save data in JSONL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} examples to {filepath}")


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=800)
    parser.add_argument('--dev_size', type=int, default=150)
    parser.add_argument('--out_dir', default='data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Generate train set
    print(f"Generating {args.train_size} training examples...")
    generator = NoisySTTDataGenerator(seed=args.seed)
    train_data = generator.generate_dataset(args.train_size, start_id=0)
    save_jsonl(train_data, os.path.join(args.out_dir, 'train.jsonl'))
    
    # Generate dev set
    print(f"Generating {args.dev_size} dev examples...")
    generator = NoisySTTDataGenerator(seed=args.seed + 1)
    dev_data = generator.generate_dataset(args.dev_size, start_id=args.train_size)
    save_jsonl(dev_data, os.path.join(args.out_dir, 'dev.jsonl'))
    
    # Print statistics
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    
    train_entities = sum(len(ex['entities']) for ex in train_data)
    dev_entities = sum(len(ex['entities']) for ex in dev_data)
    
    print(f"Train: {len(train_data)} examples, {train_entities} entities")
    print(f"Dev: {len(dev_data)} examples, {dev_entities} entities")
    
    # Entity type counts
    from collections import Counter
    train_types = Counter(e['label'] for ex in train_data for e in ex['entities'])
    
    print(f"\nEntity distribution (train):")
    for entity_type, count in sorted(train_types.items()):
        print(f"  {entity_type}: {count}")
    
    # Show samples
    print(f"\n{'='*60}")
    print("SAMPLE EXAMPLES")
    print(f"{'='*60}")
    
    for i in range(3):
        ex = train_data[i]
        print(f"\n[{ex['id']}]")
        print(f"Text: {ex['text']}")
        print(f"Entities:")
        for ent in ex['entities']:
            text_snippet = ex['text'][ent['start']:ent['end']]
            print(f"  - {ent['label']}: '{text_snippet}' [{ent['start']}:{ent['end']}]")


if __name__ == '__main__':
    main()