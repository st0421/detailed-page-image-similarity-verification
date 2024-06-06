if [ 1 -eq 1 ]; then
  # for var in 0 1 2 3 4 5 6 7 8 9 10
  # do
  #     python demo.py --query_index=$var --test_dir ../detail --mat detail_cutmix_030
  #     python demo.py --query_index=$var --test_dir ../detail --mat detail_cutmix_050
  # done

  for var in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
  do
      python demo.py --query_index=$var --test_dir ../represent --mat represent_none_030
      python demo.py --query_index=$var --test_dir ../represent --mat represent_none_020
      # python demo.py --query_index=$var --test_dir ../represent --mat represent_cutmix_050
      # python demo.py --query_index=$var --test_dir ../represent --mat represent_cutmix_100
      # python demo.py --query_index=$var --test_dir ../represent --mat represent_cutmix_200

  done
fi     