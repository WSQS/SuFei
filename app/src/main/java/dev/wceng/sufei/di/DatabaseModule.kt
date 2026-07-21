package dev.wceng.sufei.di

import android.content.Context
import androidx.room.Room
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.wceng.sufei.data.local.room.AppDatabase
import dev.wceng.sufei.data.local.room.PoemDao
import dev.wceng.sufei.data.local.room.PoetDao
import dev.wceng.sufei.data.local.room.TagDao
import dev.wceng.sufei.data.local.room.TuneDao
import dev.wceng.sufei.fork.sopho.data.local.room.AnthologyDao  // fork-specific
import dev.wceng.sufei.fork.sopho.data.local.room.AnthologyOrderingDao  // fork-specific
import dev.wceng.sufei.fork.sopho.data.local.room.ReadingProgressDao  // fork-specific
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    @Provides
    @Singleton
    fun provideAppDatabase(
        @ApplicationContext context: Context
    ): AppDatabase {
        return Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "sufei.db"
        ).addMigrations(
            AppDatabase.MIGRATION_1_2,
            AppDatabase.MIGRATION_2_3,
            AppDatabase.MIGRATION_3_4,
            AppDatabase.MIGRATION_4_5,
            AppDatabase.MIGRATION_5_6,
            AppDatabase.MIGRATION_6_7,
            AppDatabase.MIGRATION_7_8,
            AppDatabase.MIGRATION_8_9,
            AppDatabase.MIGRATION_9_10, // fork-specific: reading_progress table
            AppDatabase.MIGRATION_10_11, // fork-specific: anthologies table
            AppDatabase.MIGRATION_11_12, // fork-specific: anthology_ordering table
        )
            .build()
    }

    @Provides
    fun providePoemDao(database: AppDatabase): PoemDao {
        return database.poemDao()
    }

    @Provides
    fun providePoetDao(database: AppDatabase): PoetDao {
        return database.poetDao()
    }

    @Provides
    fun provideTagDao(database: AppDatabase): TagDao {
        return database.tagDao()
    }

    @Provides
    fun provideTuneDao(database: AppDatabase): TuneDao {
        return database.tuneDao()
    }

    // fork-specific: Reading Paths
    @Provides
    fun provideReadingProgressDao(database: AppDatabase): ReadingProgressDao {
        return database.readingProgressDao()
    }

    // fork-specific: Anthologies
    @Provides
    fun provideAnthologyDao(database: AppDatabase): AnthologyDao {
        return database.anthologyDao()
    }

    // fork-specific: Anthology original-book ordering
    @Provides
    fun provideAnthologyOrderingDao(database: AppDatabase): AnthologyOrderingDao {
        return database.anthologyOrderingDao()
    }
}
