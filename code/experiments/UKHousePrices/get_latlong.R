## This code is for the one time task of getting lat long from UK postal code and British national grid
## Output of this script is in pre-stored in our python package, one do not need to run this R script to repeat our experiments

library(sf)
all_csvs <- list.files("./CSV", pattern = ".csv", full.names = TRUE) |>
    lapply(st_read, options=c("X_POSSIBLE_NAMES=field_3",
                            "Y_POSSIBLE_NAMES=field_4")) |>
    Reduce(f = rbind) # this read in the coordinate, under British national grid system

UK_points <- st_as_sf(all_csvs, crs = st_crs(27700)) |> 
    st_set_crs(st_crs(27700)) |>
    st_transform(st_crs(4326)) # transform British national grid (27700) to lat-long (4326)

coords <- st_coordinates(UK_points) # get numeric lat long

outputs <- cbind(as.data.frame(UK_points), coords)
outputs$geometry <- NULL # remove geometry (so that the output is a bit cleaner)

names(outputs) <- c("PC",	"PQ", "EA",	"NO", 
  "CY", "RH", "LH", "CC", 
  "DC", "WC", "LONG", "LAT")

write.csv(outputs, "./combined_ukzip.csv", row.names = FALSE)
