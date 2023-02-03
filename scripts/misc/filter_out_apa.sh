#Purpose: filter out all the overlaps with known C/APA sites
bedtools intersect -v -a <cryptic_regions> -b <known CPA sites from PolyADB>
