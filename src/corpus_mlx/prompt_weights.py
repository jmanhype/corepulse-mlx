"""
Automatic1111-style prompt weighting parser for CorePulse.
Supports (word:weight) and ((word)) syntax.
"""

import re
from typing import List, Tuple, Optional


class PromptWeightParser:
    """Parser for weighted prompts using A1111 syntax."""
    
    def __init__(self):
        # Regex patterns for different weight syntaxes
        self.weight_pattern = re.compile(r'\(([^():]+):([0-9.]+)\)')  # (word:1.5)
        self.emphasis_pattern = re.compile(r'\(\(([^()]+)\)\)')  # ((word))
        self.deemphasis_pattern = re.compile(r'\[\[([^\[\]]+)\]\]')  # [[word]]
        self.alternation_pattern = re.compile(r'\[([^|\[\]]+)\|([^|\[\]]+)\]')  # [word1|word2]
        self.schedule_pattern = re.compile(r'\[([^:\[\]]+):([^:\[\]]+):([0-9.]+)\]')  # [from:to:0.5]
        
    def parse(self, prompt: str) -> Tuple[List[str], List[float]]:
        """Parse a weighted prompt into tokens and weights.
        
        Args:
            prompt: Prompt with weight syntax
            
        Returns:
            Tuple of (tokens, weights)
        """
        tokens = []
        weights = []
        
        # Work with a copy
        working_prompt = prompt
        
        # Process different patterns
        working_prompt = self._process_emphasis(working_prompt, tokens, weights)
        working_prompt = self._process_weighted(working_prompt, tokens, weights)
        working_prompt = self._process_alternation(working_prompt, tokens, weights)
        working_prompt = self._process_schedule(working_prompt, tokens, weights)
        
        # Add remaining text with default weight
        if working_prompt.strip():
            for token in working_prompt.split():
                tokens.append(token)
                weights.append(1.0)
        
        return tokens, weights
    
    def _process_emphasis(self, prompt: str, tokens: list, weights: list) -> str:
        """Process ((word)) emphasis syntax.
        
        Args:
            prompt: Input prompt
            tokens: Token list to append to
            weights: Weight list to append to
            
        Returns:
            Prompt with emphasis patterns removed
        """
        def replace_emphasis(match):
            word = match.group(1)
            tokens.append(word)
            weights.append(1.3)  # Standard emphasis multiplier
            return ""
        
        prompt = self.emphasis_pattern.sub(replace_emphasis, prompt)
        
        def replace_deemphasis(match):
            word = match.group(1)
            tokens.append(word)
            weights.append(0.7)  # Standard de-emphasis multiplier
            return ""
        
        prompt = self.deemphasis_pattern.sub(replace_deemphasis, prompt)
        
        return prompt
    
    def _process_weighted(self, prompt: str, tokens: list, weights: list) -> str:
        """Process (word:weight) syntax.
        
        Args:
            prompt: Input prompt
            tokens: Token list to append to
            weights: Weight list to append to
            
        Returns:
            Prompt with weight patterns removed
        """
        def replace_weighted(match):
            word = match.group(1)
            weight = float(match.group(2))
            tokens.append(word)
            weights.append(weight)
            return ""
        
        return self.weight_pattern.sub(replace_weighted, prompt)
    
    def _process_alternation(self, prompt: str, tokens: list, weights: list) -> str:
        """Process [word1|word2] alternation syntax.
        
        Args:
            prompt: Input prompt
            tokens: Token list to append to
            weights: Weight list to append to
            
        Returns:
            Prompt with alternation patterns processed
        """
        def replace_alternation(match):
            word1 = match.group(1)
            word2 = match.group(2)
            # For simplicity, use first word with slight emphasis
            tokens.append(word1)
            weights.append(1.1)
            return ""
        
        return self.alternation_pattern.sub(replace_alternation, prompt)
    
    def _process_schedule(self, prompt: str, tokens: list, weights: list) -> str:
        """Process [from:to:when] schedule syntax.
        
        Args:
            prompt: Input prompt
            tokens: Token list to append to
            weights: Weight list to append to
            
        Returns:
            Prompt with schedule patterns processed
        """
        def replace_schedule(match):
            from_word = match.group(1)
            to_word = match.group(2)
            when = float(match.group(3))
            
            # For simplicity, blend based on 'when' value
            if when < 0.5:
                tokens.append(from_word)
                weights.append(1.0 + (1.0 - when * 2) * 0.3)
            else:
                tokens.append(to_word)
                weights.append(1.0 + (when - 0.5) * 2 * 0.3)
            return ""
        
        return self.schedule_pattern.sub(replace_schedule, prompt)
    
    def apply_weights(
        self,
        embeddings,
        weights: List[float],
        mode: str = "multiply"
    ):
        """Apply weights to embeddings.
        
        Args:
            embeddings: Token embeddings
            weights: Weight values for each token
            mode: How to apply weights ("multiply", "add", "power")
            
        Returns:
            Weighted embeddings
        """
        import mlx.core as mx
        
        if mode == "multiply":
            # Simple multiplication
            weight_tensor = mx.array(weights).reshape(-1, 1)
            return embeddings * weight_tensor
            
        elif mode == "add":
            # Additive weighting
            weight_tensor = mx.array(weights).reshape(-1, 1)
            return embeddings + (embeddings * (weight_tensor - 1.0))
            
        elif mode == "power":
            # Power-based weighting (stronger effect)
            weight_tensor = mx.array(weights).reshape(-1, 1)
            sign = mx.sign(embeddings)
            abs_emb = mx.abs(embeddings)
            return sign * (abs_emb ** weight_tensor)
            
        else:
            return embeddings


class PromptScheduler:
    """Schedules prompt changes over denoising steps."""
    
    def __init__(self, num_steps: int):
        """Initialize scheduler.
        
        Args:
            num_steps: Total number of denoising steps
        """
        self.num_steps = num_steps
        self.schedule = []
        
    def add_prompt_change(
        self,
        from_prompt: str,
        to_prompt: str,
        start_step: int,
        end_step: Optional[int] = None
    ):
        """Add a prompt change to the schedule.
        
        Args:
            from_prompt: Starting prompt
            to_prompt: Ending prompt
            start_step: When to start transition
            end_step: When to end transition (None = until end)
        """
        if end_step is None:
            end_step = self.num_steps
            
        self.schedule.append({
            "from": from_prompt,
            "to": to_prompt,
            "start": start_step,
            "end": end_step
        })
        
    def get_prompt_at_step(self, step: int) -> str:
        """Get the interpolated prompt at a specific step.
        
        Args:
            step: Current denoising step
            
        Returns:
            Prompt for this step
        """
        for change in self.schedule:
            if change["start"] <= step < change["end"]:
                # Calculate interpolation factor
                progress = (step - change["start"]) / (change["end"] - change["start"])
                
                # For now, return appropriate prompt based on progress
                if progress < 0.5:
                    return change["from"]
                else:
                    return change["to"]
                    
        # Default to first prompt if no schedule matches
        if self.schedule:
            return self.schedule[0]["from"]
        return ""
