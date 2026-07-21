package dev.wceng.sufei.di

import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dev.wceng.sufei.data.repository.*
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
