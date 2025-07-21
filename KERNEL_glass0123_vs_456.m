MAX_GMEAN=0;
MAX_TESOVA=0;
AVG_TRAIN_TIME=0;
AVG_TEST_TIME=0;
MAX_AUC_soft=0;
MAX_AUC_hard=0;
MAX_F_MEASURE=0;
  for lambda=-18:2:50 
      
      for lambda1=-8:2:50
          
      for bias=-18:2:20
        lambda
        bias
      
         
      
             [max_gmean1, max_tesova1,X1,Y1,AUC_soft1,AUC_hard1,F_measure1,training_time1, testing_time1]= UGECSKELM('glass-0-1-2-3_vs_4-5-6-5-1tra_MM.dat','glass-0-1-2-3_vs_4-5-6-5-1tst_MM.dat', 1,'RBF_kernel',power(2,lambda),power(2,lambda1),power(2,bias));
             [max_gmean2, max_tesova2,X2,Y2,AUC_soft2,AUC_hard2,F_measure2,training_time2, testing_time2]= UGECSKELM('glass-0-1-2-3_vs_4-5-6-5-2tra_MM.dat','glass-0-1-2-3_vs_4-5-6-5-2tst_MM.dat', 1,'RBF_kernel',power(2,lambda),power(2,lambda1),power(2,bias));
             [max_gmean3, max_tesova3,X3,Y3,AUC_soft3,AUC_hard3,F_measure3,training_time3, testing_time3]= UGECSKELM('glass-0-1-2-3_vs_4-5-6-5-3tra_MM.dat','glass-0-1-2-3_vs_4-5-6-5-3tst_MM.dat', 1,'RBF_kernel',power(2,lambda),power(2,lambda1),power(2,bias));
             [max_gmean4, max_tesova4,X4,Y4,AUC_soft4,AUC_hard4,F_measure4,training_time4, testing_time4]= UGECSKELM('glass-0-1-2-3_vs_4-5-6-5-4tra_MM.dat','glass-0-1-2-3_vs_4-5-6-5-4tst_MM.dat', 1,'RBF_kernel',power(2,lambda),power(2,lambda1),power(2,bias));
             [max_gmean5, max_tesova5,X5,Y5,AUC_soft5,AUC_hard5,F_measure5,training_time5, testing_time5]= UGECSKELM('glass-0-1-2-3_vs_4-5-6-5-5tra_MM.dat','glass-0-1-2-3_vs_4-5-6-5-5tst_MM.dat', 1,'RBF_kernel',power(2,lambda),power(2,lambda1),power(2,bias));

             AMEAN_OF_MAX_GMEAN= (max_gmean1 + max_gmean2 + max_gmean3 + max_gmean4 + max_gmean5)/5;
           AMEAN_OF_MAX_TESOVA= (max_tesova1 + max_tesova2 + max_tesova3 + max_tesova4 + max_tesova5)/5;
           AMEAN_OF_MAX_AUC_soft= (AUC_soft1 +AUC_soft2 +AUC_soft3 +AUC_soft4 + AUC_soft5)/5;
           AMEAN_OF_MAX_AUC_hard= (AUC_hard1 +AUC_hard2 +AUC_hard3 +AUC_hard5 + AUC_hard5)/5;
           AMEAN_OF_MAX_F_MEASURE= (F_measure1 + F_measure2 + F_measure3 + F_measure4 + F_measure5)/5;
           AMEAN_OF_MAX_TRAIN_TIME= (training_time1 + training_time2 + training_time3 + training_time4 + training_time5)/5;
           AMEAN_OF_MAX_TEST_TIME= (testing_time1 + testing_time2 + testing_time3 + testing_time4 + testing_time5)/5;
     
       
         MEAN_OF_MAX_GMEAN=mean(AMEAN_OF_MAX_GMEAN);
         MEAN_OF_MAX_TESOVA=mean(AMEAN_OF_MAX_TESOVA);
         MEAN_OF_MAX_AUC_soft=mean(AMEAN_OF_MAX_AUC_soft);
         MEAN_OF_MAX_AUC_hard=mean(AMEAN_OF_MAX_AUC_hard);
         MEAN_OF_MAX_F_MEASURE=mean(AMEAN_OF_MAX_F_MEASURE);
         MEAN_OF_MAX_TRAIN_TIME=mean(AMEAN_OF_MAX_TRAIN_TIME);
         MEAN_OF_MAX_TEST_TIME= mean(AMEAN_OF_MAX_TEST_TIME);
         
         
         STDGMN=std(AMEAN_OF_MAX_GMEAN);
         STDTESOVA=std(AMEAN_OF_MAX_TESOVA);
          STDAUC_soft=std(AMEAN_OF_MAX_AUC_soft);
           STDAUC_hard=std(AMEAN_OF_MAX_AUC_hard);
         STDFM=std(AMEAN_OF_MAX_F_MEASURE);

            if ( MEAN_OF_MAX_GMEAN > MAX_GMEAN)
                 MAX_GMEAN=MEAN_OF_MAX_GMEAN;
                  MAX_GMEAN
                 save('KERNEL_MAX_GMEAN_glass-0-1-2-3_vs_4-5-6_1LXL1_MM')
            end

            if ( MEAN_OF_MAX_TESOVA > MAX_TESOVA)
                 MAX_TESOVA=MEAN_OF_MAX_TESOVA;
                 save('KERNEL_MAX_TESOVA_glass-0-1-2-3_vs_4-5-6_1LXL1_MM')
            end
             if ( MEAN_OF_MAX_AUC_soft > MAX_AUC_soft)
                 MAX_AUC_soft=MEAN_OF_MAX_AUC_soft;
                 save('KERNEL_MAX_AUC_soft_glass-0-1-2-3_vs_4-5-6_1LXL1_MM')
             end
             if ( MEAN_OF_MAX_AUC_hard > MAX_AUC_hard)
                 MAX_AUC_hard=MEAN_OF_MAX_AUC_hard;
                 save('KERNEL_MAX_AUC_soft_glass-0-1-2-3_vs_4-5-6_1LXL1_MM')
            end

            if ( MEAN_OF_MAX_F_MEASURE > MAX_F_MEASURE)
                 MAX_F_MEASURE=MEAN_OF_MAX_F_MEASURE;
                 save('KERNEL_MAX_F_MEASURE_glass-0-1-2-3_vs_4-5-6_1LXL1_MM')
            end
             if (  MEAN_OF_MAX_TRAIN_TIME > AVG_TRAIN_TIME)
                 AVG_TRAIN_TIME=MEAN_OF_MAX_TRAIN_TIME;
                 save(' AVG_TRAIN_TIME_glass-0-1-2-3_vs_4-5-6_1LXL1_MM')
            end

            if ( MEAN_OF_MAX_TEST_TIME > AVG_TEST_TIME)
                 AVG_TEST_TIME=MEAN_OF_MAX_TEST_TIME;
                 save('AVG_TEST_TIME_glass-0-1-2-3_vs_4-5-6_1LXL1_MM')
            end
     
      end %end bias loop
      end % end lambda1 loop
  end %end lambda loop
