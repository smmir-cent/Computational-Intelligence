# -*- coding: utf-8 -*-

# python imports
from math import degrees

# pyfuzzy imports
from fuzzy.storage.fcl.Reader import Reader
import re
import numpy as np
class FuzzyController:

    def __init__(self, fcl_path):
        self.readRules()
        self.system = Reader().load_from_file(fcl_path)


    def _make_input(self, world):
        return dict(
            cp = world.x,
            cv = world.v,
            pa = degrees(world.theta),
            pv = degrees(world.omega)
        )


    def _make_output(self):
        return dict(
            force = 0.
        )

    def _make_pa(self):
        return dict(
            up_more_right = 0.,
            up_right = 0.,
            up = 0.,
            up_left = 0.,
            up_more_left = 0.,
            down_more_left = 0.,
            down_left = 0.,
            down = 0.,
            down_right = 0.,
            down_more_right = 0.
        )


    def _make_pv(self):
        return dict(
            cw_fast = 0.,
            cw_slow = 0.,
            stop = 0.,
            ccw_slow = 0.,
            ccw_fast = 0.
        )


    def _make_force(self):
        return dict(
            left_fast = 0.,
            left_slow = 0.,
            stop = 0.,
            right_slow = 0.,
            right_fast = 0.
        )

    def readRules(self):
        self.and_rules = []
        self.or_rules = []
        with open('controllers\\rule_1.txt') as fp:
            for line in fp:
                x = re.search(r"(pa IS \w+)", line)
                x2 = re.search(r"(pv IS \w+)", line)
                x3 = re.search(r"(force IS \w+)", line)
                if "AND" in line:
                    self.and_rules.append([line[x.span()[0]+6:x.span()[1]],line[x2.span()[0]+6:x2.span()[1]],line[x3.span()[0]+9:x3.span()[1]]])
                elif "OR" in line:
                    self.or_rules.append([line[x.span()[0]+6:x.span()[1]],line[x2.span()[0]+6:x2.span()[1]],line[x3.span()[0]+9:x3.span()[1]]])
                    
        return


    def decide(self, world):
        ##  {'pa': 89.39126135805252, 'cp': 3.2716264821233403, 'pv': 0.2775948553377683, 'cv': 0.148646592901583}
        output = self._make_output()
        # self.FES(self._make_input(world), output)
        # 2 is my function
        whose_function = 2 
        if whose_function == 1: 
            self.system.calculate(self._make_input(world), output)
        elif whose_function == 2:
            self.FES(self._make_input(world), output)
        return output['force']

    

    def pa_fuzzy(self,value):
        pa = self._make_pa()
        slop = float(1)/30
        if value > 0 and value < 60: 
            if value < 30 : 
                pa['up_more_right'] = (slop) * value 
            else : 
                pa['up_more_right'] = (-slop) * value + 2
        if value > 30 and value < 90: 
            if value < 60 : 
                pa['up_right'] = (slop) * value - 1 
            else : 
                pa['up_right'] = (-slop) * value + 3
        if value > 60 and value < 120: 
            if value < 90 : 
                pa['up'] = (slop) * value - 2
            else : 
                pa['up'] = (-slop) * value + 4
        if value > 90 and value < 150: 
            if value < 120 : 
                pa['up_left'] = (slop) * value - 3 
            else : 
                pa['up_left'] = (-slop) * value + 5
        if value > 120 and value < 180: 
            if value < 150 : 
                pa['up_more_left'] = (slop) * value - 4
            else : 
                pa['up_more_left'] = (-slop) * value + 6
        if value > 180 and value < 240: 
            if value < 210 : 
                pa['down_more_left'] = (slop) * value - 6
            else : 
                pa['down_more_left'] = (-slop) * value + 8
        if value > 210 and value < 270: 
            if value < 240 : 
                pa['down_left'] = (slop) * value - 7
            else : 
                pa['down_left'] = (-slop) * value + 9
        if value > 240 and value < 300: 
            if value < 270 : 
                pa['down'] = (slop) * value - 8 
            else : 
                pa['down'] = (-slop) * value + 10
        if value > 270 and value < 330: 
            if value < 300 : 
                pa['down_right'] = (slop) * value - 9
            else : 
                pa['down_right'] = (-slop) * value + 11
        if value > 300 and value < 360: 
            if value < 330 : 
                pa['down_more_right'] = (slop) * value - 10 
            else : 
                pa['down_more_right'] = (-slop) * value + 12
        return pa


    def pv_fuzzy(self,value):
        pv = self._make_pv()
        ## cw_fast
        if value <-200 :
            pv['cw_fast'] = 1

        if value > -200 and value < -100:
            pv['cw_fast'] = (-0.01) * value - 1
        
        ## cw_slow
        if value > -200 and value < 0:
            if value < -100 :
                pv['cw_slow'] = (0.01) * value + 2
            else:
                pv['cw_slow'] = (-0.01) * value
        
        ## stop
        if value > -100 and value < 100:
            if value < 0 :
                pv['stop'] = (0.01) * value + 1
            else:
                pv['stop'] = (-0.01) * value + 1      
        
        ## ccw_slow
        if value > 0 and value < 200:
            if value < 100 :
                pv['ccw_slow'] = (0.01) * value 
            else:
                pv['ccw_slow'] = (-0.01) * value + 2            
        
        ## ccw_fast
        if value > 100 and value < 200:
            pv['ccw_fast'] = (0.01) * value - 1     

        if value > 200  :
            pv['ccw_fast'] = 1

        return pv



    def left_fast_force(self,value,threshold): 
        if value > -100 and value < -60: 
            if value <-80: 
                return min(((0.05*value) + 5 ),threshold) 
            else: 
                return min(((-0.05*value) - 3),threshold)
        return 0.


    def left_slow_force(self,value,threshold): 
        if value > -80 and value < 0: 
            if value <-60: 
                return min(((0.05*value) + 4),threshold)
            else: 
                return min((((float(-1)/60)*value)),threshold)
        return 0.



    def stop_force(self,value,threshold): 
        if value > -60 and value < 60: 
            if value < 0: 
                return min((((float(1)/60)*value) + 1 ),threshold)
            else: 
                return min((((float(-1)/60)*value) + 1),threshold)
        return 0.


    def right_slow_force(self,value,threshold): 
        if value > 0 and value < 80: 
            if value < 60: 
                return min((((float(1)/60)*value) + 5 ),threshold)
            else: 
                return min(((-0.05*value) + 4),threshold)
        return 0.



    def right_fast_force(self,value,threshold): 
        if value > 60 and value < 100: 
            if value < 80: 
                return min(((0.05*value) - 3),threshold)
            else: 
                return min(((-0.05*value) + 5),threshold)
        return 0.


    def fuzzify(self,input):
        input_pa = self.pa_fuzzy(input.get('pa'))
        input_pv = self.pv_fuzzy(input.get('pv'))
        return [input_pa,input_pv]


    def rules(self,fuzzy_input):
        force = self._make_force()
        for and_rule in self.and_rules:
            temp = min(fuzzy_input[0][and_rule[0]],fuzzy_input[1][and_rule[1]])
            force[and_rule[2]] = max(temp,force[and_rule[2]]) 
        for or_rule in self.and_rules:
            temp = min(fuzzy_input[0][or_rule[0]],fuzzy_input[1][or_rule[1]])
            force[or_rule[2]] = max(temp,force[or_rule[2]])              
        return force


    def defuzzify_force(self,force):
        points = np.linspace(-100,100,50)
        dx = points[1] - points[0]
        sum = 0.
        sum_2 = 0.
        for point in points:
            y = max(self.stop_force(point,force['stop']),self.left_fast_force(point,force['left_fast']),self.left_slow_force(point,force['left_slow']),self.right_slow_force(point,force['right_slow']),self.right_fast_force(point,force['right_fast']))
            sum += (y*dx*point)
            sum_2 +=  (y*dx)
        print("sum = ")
        print(sum)
        print("sum_2 = ")
        print(sum_2)
        if sum == 0 or sum_2 == 0:
            return 0.
        return sum/sum_2


    def FES(self,input, output):
        fuzzy_input = self.fuzzify(input=input)
        force = self.rules(fuzzy_input=fuzzy_input)
        # self.defuzzify_force(force=force)
        output['force'] = self.defuzzify_force(force=force)
        print(output['force'])
        return
        
