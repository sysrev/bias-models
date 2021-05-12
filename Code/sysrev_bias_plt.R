##%%%%%%%%%%
## Sysrev bias plotting
##%%%%%%%%%%

rm(list = ls())


# Libraries
library(ggplot2)

# Main

# Load data
ento_lime_df <- read.csv("../Results/ento_lime.csv", stringsAsFactors = F)
ento_lr_feat_count <- read.csv("../Results/ento_lr_count.csv", stringsAsFactors = F)
ento_lr_feat_tfidf <- read.csv("../Results/ento_lr_tfidf.csv", stringsAsFactors = F)

clim_lime <- read.csv("../Results/clim_lime.csv", stringsAsFactors = F)
clim_lr_feat_count <- read.csv("../Results/clim_lr_count.csv", stringsAsFactors = F)
clim_lr_feat_tfidf <- read.csv("../Results/clim_lr_tfidf.csv", stringsAsFactors = F)

frag_lime <- read.csv("../Results/frag_lime.csv", stringsAsFactors = F)
frag_lr_feat_count <- read.csv("../Results/frag_lr_count.csv", stringsAsFactors = F)
frag_lr_feat_tfidf <- read.csv("../Results/frag_lr_tfidf.csv", stringsAsFactors = F)


# get top and bottom 25 terms
# term, value for lime
# Feature, Coefficient for lr


# 

# Plotting..

df_prep_lr <- function(df, dat){
    tmp_top <- df[1:25,]
    tmp_bot <- df[(nrow(df)-24):nrow(df),]
    
    tmp_top <- tmp_top[order(-tmp_top$Coefficient),]
    tmp_top$x <- 1:nrow(tmp_top)
    
    tmp_bot <- tmp_bot[order(tmp_bot$Coefficient),]
    tmp_bot$x <- 1:nrow(tmp_bot)
    
    tmp <- rbind(tmp_top, tmp_bot)
    tmp$sign <- sign(tmp$Coefficient)
    # tmp$lab_y <- (tmp$sign * max(abs(tmp$Coefficient))) + (tmp$sign * 0.75)
    # tmp$sign <- as.character(tmp$sign)
    
    tmp$Data <- dat
    
    tmp
}

df_prep_lime <- function(df, dat){
    tmp_top <- df[order(-df$value),][1:25,]
    tmp_bot <- df[order(df$value),][1:25,]
    
    # tmp_top <- tmp_top[order(-tmp_top$Coefficient),]
    tmp_top$x <- 1:nrow(tmp_top)
    
    # tmp_bot <- tmp_bot[order(tmp_bot$Coefficient),]
    tmp_bot$x <- 1:nrow(tmp_bot)
    
    tmp <- rbind(tmp_top, tmp_bot)
    tmp$sign <- sign(tmp$value)
    # tmp$lab_y <- (tmp$sign * max(abs(tmp$Coefficient))) + (tmp$sign * 0.75)
    # tmp$sign <- as.character(tmp$sign)
    
    tmp$Data <- dat
    
    tmp
}

lr_count <- rbind(df_prep_lr(ento_lr_feat_count, "1. EntoGEM"),
                  df_prep_lr(clim_lr_feat_count, "2. Climate:health"),
                  df_prep_lr(frag_lr_feat_count, "3. Frag:mammals"))

lr_count$lab_y <- (lr_count$sign * max(abs(lr_count$Coefficient))) + (lr_count$sign * 0.75)
lr_count$sign <- as.character(lr_count$sign)

lr_count$Data <- factor(lr_count$Data, 
                        levels = c("1. EntoGEM", "2. Climate:health", "3. Frag:mammals"))


lime <- rbind(df_prep_lime(ento_lime_df, "1. EntoGEM"),
              df_prep_lime(clim_lime, "2. Climate:health"),
              df_prep_lime(frag_lime, "3. Frag:mammals"))

lime$lab_y <- (lime$sign * max(abs(lime$value))) + (lime$sign * 0.75)
lime$sign <- as.character(lime$sign)

lime$Data <- factor(lime$Data, 
                        levels = c("1. EntoGEM", "2. Climate:health", "3. Frag:mammals"))


col <- c("firebrick3", "dodgerblue3")
names(col) <- c("-1", "1")

basic_thm <- theme(axis.text = element_text(size = 16),
                   axis.title = element_text(size = 20),
                   plot.title = element_text(size = 20),
                   plot.subtitle = element_text(size = 18),
                   strip.text.x = element_text(size = 18),
                   strip.text.y = element_text(size = 18),
                   strip.background = element_blank(),
                   legend.text = element_text(size = 16),
                   legend.title = element_text(size = 18),
                   legend.text.align = 0) 

lr_plt <- ggplot(lr_count) +
    geom_hline(aes(yintercept = 0), lty = 1, colour = "grey") +
    geom_point(aes(x = -x, y = Coefficient, colour = sign),
               alpha = 0.9,
               size = 2.5,
               show.legend = F) +
    geom_segment(aes(x = -x, xend = -x, yend = 0, y = Coefficient, colour = sign),
                 size = 1.5,
                 alpha = 0.9,
                 show.legend = F) +
    scale_colour_manual(values = col) +
    # geom_histogram(aes(x = -x, y = Coefficient, fill = sign), colour = NA,
    #                stat = "identity", position = "identity", alpha = 0.9,
    #                show.legend = F) +
    # scale_fill_manual(values = col) +
    geom_text(aes(x = -x, y = lab_y, label = Feature),
              hjust = 0,
              data = subset(lr_count, Coefficient < 0)) +
    geom_text(aes(x = -x, y = lab_y, label = Feature),
              hjust = 1,
              data = subset(lr_count, Coefficient > 0)) +
    ylab("Weight") +
    ggtitle("Text-based associations") +
    coord_flip() +
    theme_bw() +
    basic_thm + 
    theme(axis.line.y = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          panel.grid = element_blank()) +
    facet_grid(~Data)

lime_plt <- ggplot(lime) +
    geom_hline(aes(yintercept = 0), lty = 1, colour = "grey") +
    # geom_histogram(aes(x = -x, y = value, fill = sign), colour = NA,
    #                stat = "identity", position = "identity", alpha = 0.9,
    #                show.legend = F) +
    geom_point(aes(x = -x, y = value, colour = sign),
                   alpha = 0.9,
               size = 2.5,
                   show.legend = F) +
    geom_segment(aes(x = -x, xend = -x, yend = 0, y = value, colour = sign),
                 size = 1.5,
               alpha = 0.9,
               show.legend = F) +
    scale_colour_manual(values = col) +
    geom_text(aes(x = -x, y = lab_y, label = term),
              hjust = 0,
              data = subset(lime, value < 0)) +
    geom_text(aes(x = -x, y = lab_y, label = term),
              hjust = 1,
              data = subset(lime, value > 0)) +
    ylab("Weight") +
    ggtitle("Model-based associations") +
    coord_flip() +
    theme_bw() +
    basic_thm + 
    theme(axis.line.y = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          panel.grid = element_blank()) +
    facet_grid(~Data)

ggsave("../Figs/lr_plt.pdf", # Results/
       lr_plt,
       device = "pdf", dpi = 300,
       width = 15, height = 7.5)

ggsave("../Figs/lime_plt.pdf",
       lime_plt,
       device = "pdf", dpi = 300,
       width = 15, height = 7.5)
