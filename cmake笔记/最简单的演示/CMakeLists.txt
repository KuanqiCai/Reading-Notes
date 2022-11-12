cmake_minimum_required(VERSION 3.4.3)
 
macro(arg_test para1 para2)
        MESSAGE(STATUS ARGC=${ARGC})
        MESSAGE(STATUS ARGV=${ARGV})
        MESSAGE(STATUS ARGN=${ARGN})
        MESSAGE(STATUS ARGV0=${ARGV0})
        MESSAGE(STATUS ARGV1=${ARGV1})
        MESSAGE(STATUS ARGV2=${ARGV2})
endmacro()
 
arg_test(para_1, para_2, para_3, para_4)