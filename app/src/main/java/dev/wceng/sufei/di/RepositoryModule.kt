package dev.wceng.sufei.di

import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dev.wceng.sufei.data.repository.*
import dev.wceng.sufei.fork.sopho.data.repository.ForkImportRepository  // fork-specific
import dev.wceng.sufei.fork.sopho.data.repository.ForkImportRepositoryImpl  // fork-specific
import dev.wceng.sufei.fork.sopho.data.repository.ReadingPathRepository  // fork-specific
import dev.wceng.sufei.fork.sopho.data.repository.ReadingPathRepositoryImpl  // fork-specific
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Binds
    @Singleton
    abstract fun bindPoemRepository(
        poemRepositoryImpl: PoemRepositoryImpl
    ): PoemRepository

    @Binds
    @Singleton
    abstract fun bindImportRepository(
        importRepositoryImpl: ImportRepositoryImpl
    ): ImportRepository

    @Binds
    @Singleton
    abstract fun bindUserPreferencesRepository(
        userPreferencesRepositoryImpl: UserPreferencesRepositoryImpl
    ): UserPreferencesRepository

    // fork-specific: Reading Paths
    @Binds
    @Singleton
    abstract fun bindReadingPathRepository(
        readingPathRepositoryImpl: ReadingPathRepositoryImpl
    ): ReadingPathRepository

    // fork-specific: fork data import (anthologies, etc.)
    @Binds
    @Singleton
    abstract fun bindForkImportRepository(
        forkImportRepositoryImpl: ForkImportRepositoryImpl
    ): ForkImportRepository
}
