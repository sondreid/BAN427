# Libraries --------------------
library(readxl)   
library(fs)
library(openxlsx)
library(tidyverse)
library(fable)
library(dplyr)
library(openxlsx)
library(magrittr)
library(Cairo)
library(ggplot2)
library(lubridate)
library(scales)
library(caret)



df <- read_excel("exam_case_data.xlsx")

colnames(df)


df %>% 
  filter((TENURE_TIME2-TENURE_TIME1) == 0.5) %>% nrow()

  
df %>% 
  filter((TENURE_TIME2-TENURE_TIME1) == 0.5 | is.na(TIME2)) %>% nrow()
