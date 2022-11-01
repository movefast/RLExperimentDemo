"""An abstract class that specifies a minial agent interface.
"""
from abc import ABCMeta, abstractmethod


class AbstractBaseAgent:
    """Implements the agent
    Note:
        agent_init, agent_start, agent_step, agent_end are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

