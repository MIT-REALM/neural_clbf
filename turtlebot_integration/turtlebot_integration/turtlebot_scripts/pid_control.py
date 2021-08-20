# class for implementing PID control on the turtlebot
# evaluates a PID controller given current state

import time


class pid_controller(object):

    def __init__(self, p_gain, d_gain, i_gain):
        self._set_gains(p_gain, d_gain, i_gain)
        self._reset()
        self.current_time = time.time()

    def _reset(self):
        self._p_error_last = 0
        self._p_error = 0
        self._d_error = 0
        self._i_error = 0
        self._cmd = 0
        self._last_time = None
    
    
    def _set_gains(self, p_gain, d_gain, i_gain):
        """

        Setter for proportional, derivative, and integral components
        of a the controller with a built-in check for integral gain bounds        
        
        """

        self._p_gain = p_gain
        self._d_gain = d_gain
        self._i_gain = i_gain

    
    def _get_gains(self):
        return (self._p_gain, self._d_gain, self._i_gain)
        
    def _set_errors(self, current_state, goal, dt):
        """
        
        Compare current state to goal state and set errors
        accordingly for each type of gain

        """
        self._p_error = goal - current_state

        if dt > 0:
            self._d_error = (self._p_error - self._p_error_last)/dt
        else:
            self._d_error = 0
        
        self._i_error += (goal - current_state)*dt
    
    
    def _get_errors(self):
        return (self._p_error, self._d_error, self._i_error)

    
    def _previous_time(self):
        return self._last_time

    def _evaluate_controller(self, p_gain, d_gain, i_gain, p_error, d_error, i_error):
        """
        
        Output PID controller based on current state and gains

        """
        return p_gain * p_error + d_gain * d_error + i_gain * i_error

    #this is the only function that needs to be called from outside this class
    def get_command(self, current_state, goal):
        """
        
        Creates a PID controller and returns a command 
        
        """

        if self._last_time is None:
            self._last_time = time.time()
            self.current_time = self._last_time
        
        else:
            self.current_time = time.time()
        
        dt = self.current_time - self._last_time
        self._set_errors(current_state, goal, dt)
        p_error, d_error, i_error = self._get_errors()
        p_gain, d_gain, i_gain = self._get_gains()

        self.command = self._evaluate_controller(p_gain, d_gain, i_gain, p_error, d_error, i_error)


        self._last_time = self.current_time

        return self.command

        

        
        
    

    

    
    

